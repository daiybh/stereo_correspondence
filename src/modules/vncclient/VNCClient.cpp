/*!
 * @file 		VNCClient.cpp
 * @author 		Zdenek Travnicek
 * @date 		20.12.2011
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2011 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "VNCClient.h"
#include "yuri/core/Module.h"
#include "yuri/core/socket/StreamSocketGenerator.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/RawVideoFrame.h"

#include <cassert>
namespace yuri {

namespace vnc {

IOTHREAD_GENERATOR(VNCClient)
MODULE_REGISTRATION_BEGIN("vncclient")
	REGISTER_IOTHREAD("vncclient",VNCClient)
MODULE_REGISTRATION_END()

core::Parameters VNCClient::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("Receives data from VNC server");
	p["port"]["Port to connect to"]=5900;
	p["address"]["Address to connect to"]="127.0.0.1";
	return p;
}

VNCClient::VNCClient(const log::Log &log_,core::pwThreadBase parent, const core::Parameters &parameters)
:IOThread(log_,parent,0,1,"VNCClient"),buffer_size(104857600),state(awaiting_data),
	remaining_rectangles(0)
{
	IOTHREAD_INIT(parameters)
//	buffer.reset(new uint8_t[buffer_size]);
	buffer.resize(buffer_size);
	buffer_pos = &buffer[0];
	buffer_end = buffer_pos + buffer_size;
	buffer_free = buffer_size;
	buffer_valid = 0;

	socket_ = core::StreamSocketGenerator::get_instance().generate("yuri_tcp", log);
	log[log::info] << "Created socket";
}

VNCClient::~VNCClient() noexcept
{
}

void VNCClient::run()
{
	IOThread::print_id();
	if (!connect()) {
		log[log::error] << "Failed to connect";
		return;
	}
	yuri::size_t read;
	bool need_data = true;
	while (still_running()) {
		if (!buffer_valid || need_data) {
			if (socket_->wait_for_data(get_latency())) {
				read = socket_->receive_data(buffer_pos+buffer_valid,buffer_free);
				log[log::info] << "Read " << read << " bytes";
				buffer_valid+=read;
				buffer_free-=read;
				if (read) {
					need_data=false;
					last_read = timestamp_t{};//boost::posix_time::microsec_clock::local_time();
				}
			} /*else {
				IOThread::sleep(get_latency());
			}*/
		}
		if (!buffer_free) {
			log[log::error] << "Buffer full. Moving data.";
			yuri::size_t free_beg = buffer_pos - &buffer[0];
			yuri::size_t remaining = buffer_valid;
			uint8_t *pos_d = &buffer[0], *pos_s = buffer_pos;
			yuri::size_t cop;
			while (remaining) {
				cop = free_beg>=remaining?remaining:free_beg;
				//memcpy(pos_d,pos_s,cop);
				std::copy(pos_s, pos_s+cop, pos_d);
				remaining-=cop;
				pos_d+=cop;
				pos_s+=cop;
			}
			buffer_pos = &buffer[0];
			buffer_free = buffer_size - buffer_valid;
			log[log::error] << "Moved buffer " << free_beg << " bytes back";
		}
		// TODO reimplement this (?)
		timestamp_t now;// = boost::posix_time::microsec_clock::local_time();
		if (now - last_read > 100_ms) {
			request_rect(0,0,width,height,true);
			last_read = now;
		}
		if (!process_data()) need_data=true;
		if (buffer_valid==0) {
			buffer_pos = &buffer[0];
			buffer_free=buffer_size;
		}
	}

}
/** \brief Reads Connects to a remote server
 *  \return true if connection was successful, false otherwise
 */
bool VNCClient::connect()
{
	//if (!socket) socket.reset(new asio::ASIOTCPSocket(log,get_this_ptr()));
	//socket->set_endpoint(address,port);
	log[log::info] << "connecting to " << address << ":" << port;
	if (!socket_->connect(address,port)) return false;
	log[log::info] << "Connected to " << address << ":" << port;;
	if (!handshake()) return false;
	return true;
}

bool VNCClient::set_param(const core::Parameter &p)
{
	if (iequals(p.get_name(), "address")) {
		address = p.get<std::string>();
	} else if (iequals(p.get_name(), "port")) {
		port=p.get<uint16_t>();
	} else return IOThread::set_param(p);
	return true;
}
/** \brief Goes thru the handshake with the server, setting up necessary parameters
 *  \return true if handshake ended successfully, false otherwise
 */
bool VNCClient::handshake()
{
	assert(buffer_free >=13);
	size_t read = 0;
	read = read_data_at_least(buffer_pos, 12, 12);

	assert(read==12);
	buffer_pos[12]=0;
	log[log::info] << "Connected to server version: " << reinterpret_cast<char*>(buffer_pos);
	socket_->send_data(buffer_pos,12);
	read = read_data_at_least(buffer_pos, buffer_free, 1);
//	while ((read = socket_->receive_data(buffer_pos,buffer_free)) == 0) {}
	assert(read>0);
	bool plain_found = false;

	for (uint8_t i=0;i<buffer_pos[0];++i ) {

		if (buffer_pos[i+1]==1) {
			plain_found = true;
			break;
		}
	}
	if (!plain_found) {
		log[log::warning] << "No plain encoding";
		return false;
	}
	buffer_pos[0]=1;
	socket_->send_data(buffer_pos,1);
	read = read_data_at_least(buffer_pos, 4, 4);
//	read = 	socket_->receive_data(buffer_pos,4);
	assert(read==4);
	if (*reinterpret_cast<uint32_t*>(buffer_pos) != 0) {
		log[log::error] << "handshake unsuccessful";
		return false;
	}
	buffer_pos[0]=1;
	socket_->send_data(buffer_pos,1);
	read = read_data_at_least(buffer_pos, buffer_free, 24);
//	read = 	socket_->receive_data(buffer_pos,buffer_free);
	assert(read>=24);
	width = get_ushort(buffer_pos);
	height = get_ushort(buffer_pos+2);
	pixel_format.bpp = buffer_pos[4];
	pixel_format.depth = buffer_pos[5];
	for (uint16_t i = 0; i < 3; ++i) {
		pixel_format.colors[i].max = get_ushort(buffer_pos+8+i*2);
		pixel_format.colors[i].shift = *(buffer_pos+14+i);
	}
	uint32_t name_len = get_uint(buffer_pos+20);
	std::string name(reinterpret_cast<const char*>(buffer_pos+24),name_len);
	log[log::info] << "handshake finished, connected to server " << name <<
			", with resolution " << width << "x" << height;
	log[log::info] << "Server encoding uses " << pixel_format.bpp <<
			" bit per pixel, with " << pixel_format.depth <<
			" valid bits";
	log[log::info] << "Color parameters: " <<
			"red @" << static_cast<uint16_t>(pixel_format.colors[0].shift) << ", max: " <<  pixel_format.colors[0].max <<
			"green @" << static_cast<uint16_t>(pixel_format.colors[1].shift) << ", max: " <<  pixel_format.colors[1].max <<
			"blue @" << static_cast<uint16_t>(pixel_format.colors[2].shift) << ", max: " <<  pixel_format.colors[2].max;
	//image.reset(new uint8_t[width*height*3]);
	image.resize(width*height*3);
	request_rect(0,0,width,height,false);
	set_encodings();
	//enable_continuous();
	return true;
}
/** \brief Processes data in a buffer
 *
 *  The method is basically a state machine (currently with 2 distinct states)
 *  that processes data stored in a buffer a reacts to events or requests more data
 *  if there's not enough data to process
 *  \todo The method should throw an exception when unknown message type arrives.
 *
 *  \return true if some event was processed, false if more data are needed.
 */
bool VNCClient::process_data()
{
	if (!buffer_valid) return false;
	//uint16_t rectangles;
	std::string text;
	uint32_t len;
	switch (state) {
		case awaiting_data:
			log[log::debug] << "Processing data, starting with " << static_cast<uint32_t>(buffer_pos[0]) << ", valid: " << buffer_valid<< std::endl;
			switch(buffer_pos[0]) {
				case 0:if (buffer_valid < 4) return false;
					remaining_rectangles = get_ushort(buffer_pos+2);
					log[log::debug] << "Rectangles: " << remaining_rectangles;
					move_buffer(4);
					state = receiving_rectangles;
					//buffer_r
					return true;
				case 2: log[log::info] << "BELL";
					move_buffer(1);return true;
				case 3: if (buffer_valid < 8) return false;
					len = get_uint(buffer_pos+4);
					if (buffer_valid < 8+len) return false;
					text = std::string(reinterpret_cast<const char*>(buffer_pos+8),len);
					log[log::info] << "Server cut text: " << text;
					move_buffer(8+len);
					return true;
				case 150: if (buffer_valid < 10) return false;
					log[log::info] << "Continuous updates enabled";
					move_buffer(10);
					return true;
				default:
					return true;
			} break;
		case receiving_rectangles:
			if (!remaining_rectangles) {
				state = awaiting_data;
				return true;
			}
			if (buffer_valid < 12) return false;
			{
				uint16_t x,y,w,h;
				uint32_t  enc;
				x = get_ushort(buffer_pos+0);
				y = get_ushort(buffer_pos+2);
				w = get_ushort(buffer_pos+4);
				h = get_ushort(buffer_pos+6);
				enc = get_uint(buffer_pos+8);
				log[log::debug] << "Got update for rect " << w<<"x"<<h<<" at " <<x<<"x"<<y<<", encoding: " << enc;
				size_t need = 0;
				switch (enc) {
					case 0: need = w*h*(pixel_format.bpp>>3);
						break;
					case 1: need = 4;
						log[log::warning] << "copy rect!";
						break;
					default:
						log[log::error] << "Unsupported encoding!!";
				}

				if (buffer_valid < 12 + need) {
					log[log::debug] << "Not enough data (need: " << need << ", have: " << buffer_valid << "@ "<< (buffer_pos - &buffer[0])<< ")";
					return false;
				}
				switch (enc) {
					case 0: {
						uint8_t * pos = buffer_pos + 12, *outpos;
						yuri::size_t pixel;
						assert(pixel_format.bpp == 32);
						for (uint16_t line=y; line < y+h; ++line) {
							outpos = &image[0]+3*width*line+x*3;
							for (uint16_t row = x; row < x + w; ++row) {
								pixel = get_pixel(pos);
								*outpos++ = (pixel>>pixel_format.colors[2].shift)&pixel_format.colors[2].max;
								*outpos++ = (pixel>>pixel_format.colors[1].shift)&pixel_format.colors[1].max;
								*outpos++ = (pixel>>pixel_format.colors[0].shift)&pixel_format.colors[0].max;
								pos += pixel_format.bpp >> 3;
							}
						}; }
						break;
					case 1: {
						uint16_t x0 = get_ushort(buffer_pos + 12);
						uint16_t y0 = get_ushort(buffer_pos + 14);
						uint8_t *tmp, *pos=0;
						tmp = new uint8_t[w*h*3];
						uint8_t *b=tmp;
						for (uint16_t line=y0; line < y0+h; ++line) {
							pos = &image[0]+3*width*line+x0*3;
							//memcpy(b,pos,w*3);
							std::copy(pos,pos+w*3, b);
							b+=w*3;
						}
						b = tmp;
						for (uint16_t line=y; line < y+h; ++line) {
							pos = &image[0]+3*width*line+x*3;
//							memcpy(pos,b,w*3);
							std::copy(b,b+w*3, pos);
							b+=w*3;
						}
						delete [] tmp;

						} break;

					default:
						log[log::error] << "Unimplemented encoding ";
						break;
				}
				move_buffer(12+need);
				if (!--remaining_rectangles) {
					state = awaiting_data;
//					core::pBasicFrame frame = IOThread::allocate_frame_from_memory(image,width*height*3);
//					core::pBasicFrame frame = IOThread::allocate_frame_from_memory(image);
					core::pRawVideoFrame frame = core::RawVideoFrame::create_empty(core::raw_format::rgb24, {width, height}, image.data(), width*height*3,  true);
					push_frame(0,frame);
					request_rect(0,0,width,height,true);
				}
			}
			break;
	}
	return true;
}
/** \brief moves actual position in buffer
 *  \param offset offset in bytes
 */
void VNCClient::move_buffer(yuri::ssize_t offset)
{
	buffer_pos+=offset;
	buffer_valid-=offset;
}
/** \brief Reads unsigned 4B integer from a buffer
 *  \param start start of a buffer
 *  \return retrieved integer
 */
uint32_t VNCClient::get_uint(uint8_t *start)
{
	return (start[0]<<24) |
			(start[1]<<16) |
			(start[2]<<8) |
			(start[3]<<0);

}
/** \brief Reads unsigned 2B integer from a buffer
 *  \param start start of a buffer
 *  \return retrieved integer
 */
uint16_t VNCClient::get_ushort(uint8_t *start)
{
	return (start[0]<<8) |
		   (start[1]<<0);

}
/** \brief Reads a pixel value from a buffer
 *
 * Currently this method supports only 24b pixel depths, in 24b or 32b per pixel
 *  \param buf Start of a buffer
 *  \return Retrieved pixel value
 */
yuri::size_t VNCClient::get_pixel(uint8_t *buf)
{
	//uint16_t b = 0;
	yuri::size_t px = 0;
	/*while (b < pixel_format.depth) {
		px=px<<8+*buf++;
		b+=8;
	}*/
	px=(buf[0]<<16)+(buf[1]<<8)+buf[2];
	return px;
}
/** \brief Stores unsigned 2B integer to a buffer
 *  \param start Start of a buffer
 *  \param data Value to store
 */
void VNCClient::store_ushort(uint8_t *start, uint16_t data)
{
	start[0]=(data>>8)&0xFF;
	start[1]=data&0xFF;
}
void VNCClient::store_uint(uint8_t *start, uint32_t data)
{
	start[0]=(data>>24)&0xFF;
	start[1]=(data>>16)&0xFF;
	start[2]=(data>>8)&0xFF;
	start[3]=data&0xFF;
}
/** \brief Sends a request for an update of arbitrary rectangle
 *
 *  \param x Offset from the left
 *  \param y Offset from the top
 *  \param w The width of the rectangle
 *  \param h The height of the rectangle
 *  \param incremental Request incremental updates.
 *  If incremental updates are specified, the server should send only areas of the
 *  framebuffer, that changed since last framebuffer update
 *  \return Always true
 */
bool VNCClient::request_rect(uint16_t x, uint16_t y, uint16_t w, uint16_t h, bool incremental)
{
	uint8_t b[10];
	b[0]=3;
	b[1]=static_cast<uint8_t>(incremental);
	store_ushort(b+2,x);
	store_ushort(b+4,y);
	store_ushort(b+6,w);
	store_ushort(b+8,h);
	yuri::size_t size = socket_->send_data(b,10);
	assert (size==10);
	return true;
}
/** \brief Sends a request for a continuous updates
 *  \return Always true
 */

bool VNCClient::enable_continuous()
{
	uint8_t b[10];
	b[0]=150;
	b[1]=0;
	store_ushort(b+2,0);
	store_ushort(b+4,0);
	store_ushort(b+6,width);
	store_ushort(b+8,height);
	yuri::size_t size = socket_->send_data(b,10);
	assert (size==10);
	return true;
}
bool VNCClient::set_encodings()
{
	uint8_t b[12];
	b[0]=2;
	b[1]=0;
	store_ushort(b+2,2);
	store_uint(b+4,0);
	store_uint(b+8,1);
	yuri::size_t size = socket_->send_data(b,12);
	assert (size==12);
	return true;
}

size_t VNCClient::read_data_at_least(uint8_t* data, size_t size, size_t at_least)
{
	size_t read = 0;
	size_t total_read = 0;
	if (size < at_least) return 0;
	while (total_read < at_least && still_running()) {
		read = socket_->receive_data(data+total_read,size-total_read);
		total_read += read;
	}
	return total_read;
}
}

}
