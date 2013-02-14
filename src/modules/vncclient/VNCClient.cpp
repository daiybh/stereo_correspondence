/*
 * VNCClient.cpp
 *
 *  Created on: Dec 20, 2011
 *      Author: neneko
 */

#include "VNCClient.h"

namespace yuri {

namespace io {

REGISTER("vncclient",VNCClient)


IO_THREAD_GENERATOR(VNCClient)

using boost::iequals;
shared_ptr<Parameters> VNCClient::configure()
{
	shared_ptr<Parameters> p = BasicIOThread::configure();
	p->set_description("Receives data from VNC server");
	(*p)["port"]["Port to connect to"]=5900;
	(*p)["address"]["Address to connect to"]=std::string("localhost");
	return p;
}

VNCClient::VNCClient(Log &log_,pThreadBase parent, Parameters &parameters) IO_THREAD_CONSTRUCTOR:
	BasicIOThread(log_,parent,0,1,"VNCClient"),buffer_size(104857600),state(awaiting_data),
	remaining_rectangles(0)
{
	IO_THREAD_INIT("VNCClient")
	buffer.reset(new yuri::ubyte_t[buffer_size]);
	buffer_pos = buffer.get();
	buffer_end = buffer_pos + buffer_size;
	buffer_free = buffer_size;
	buffer_valid = 0;
}

VNCClient::~VNCClient()
{
}

void VNCClient::run()
{
	BasicIOThread::print_id();
	if (!connect()) return;
	yuri::size_t read;
	bool need_data = true;
	while (still_running()) {
		if (!buffer_valid || need_data) {
			if (socket->available()) {
				read = socket->read(buffer_pos+buffer_valid,buffer_free);
				log[info] << "Read " << read << " bytes" << std::endl;
				buffer_valid+=read;
				buffer_free-=read;
				if (read) {
					need_data=false;
					last_read = boost::posix_time::microsec_clock::local_time();
				}
			} else {
				BasicIOThread::sleep(latency);
			}
		}
		if (!buffer_free) {
			log[error] << "Buffer full. Moving data." << std::endl;
			yuri::size_t free_beg = buffer_pos - buffer.get();
			yuri::size_t remaining = buffer_valid;
			yuri::ubyte_t *pos_d = buffer.get(), *pos_s = buffer_pos;
			yuri::size_t cop;
			while (remaining) {
				cop = free_beg>=remaining?remaining:free_beg;
				memcpy(pos_d,pos_s,cop);
				remaining-=cop;
				pos_d+=cop;
				pos_s+=cop;
			}
			buffer_pos = buffer.get();
			buffer_free = buffer_size - buffer_valid;
			log[error] << "Moved buffer " << free_beg << " bytes back"<< std::endl;
		}
		now = boost::posix_time::microsec_clock::local_time();
		if (now-last_read > boost::posix_time::milliseconds(200)) {
			request_rect(0,0,width,height,true);
			last_read = now;
		}
		if (!process_data()) need_data=true;
		if (buffer_valid==0) {
			buffer_pos = buffer.get();
			buffer_free=buffer_size;
		}
	}

}
/** \brief Reads Connects to a remote server
 *  \return true if connection was successful, false otherwise
 */
bool VNCClient::connect()
{
	if (!socket) socket.reset(new ASIOTCPSocket(log,get_this_ptr()));
	//socket->set_endpoint(address,port);
	socket->connect(address,port);
	log[info] << "Connected to " << address << ":" << port << std::endl;
	if (!handshake()) return false;
	return true;
}

bool VNCClient::set_param(Parameter &p)
{
	if (iequals(p.name, "address")) {
		address = p.get<std::string>();
	} else if (iequals(p.name, "port")) {
		port=p.get<yuri::ushort_t>();
	} else return BasicIOThread::set_param(p);
	return true;
}
/** \brief Goes thru the handshake with the server, setting up necessary parameters
 *  \return true if handshake ended successfully, false otherwise
 */
bool VNCClient::handshake()
{
	assert(buffer_free >=13);
	yuri::size_t read = socket->read(buffer_pos,12);
	assert(read==12);
	buffer_pos[12]=0;
	log[info] << "Connected to server version: " << reinterpret_cast<char*>(buffer_pos) << std::endl;
	socket->write(buffer_pos,12);
	read = socket->read(buffer_pos,buffer_free);
	assert(read>0);
	bool plain_found = false;
	for (yuri::ubyte_t i=0;i<buffer_pos[0];++i ) {
		if (buffer_pos[i+1]==1) {
			plain_found = true;
			break;
		}
	}
	if (!plain_found) return false;
	buffer_pos[0]=1;
	socket->write(buffer_pos,1);
	read = 	socket->read(buffer_pos,4);
	assert(read==4);
	if (*reinterpret_cast<yuri::uint_t*>(buffer_pos) != 0) {
		log[error] << "handshake unsuccessful" << std::endl;
		return false;
	}
	buffer_pos[0]=1;
	socket->write(buffer_pos,1);
	read = 	socket->read(buffer_pos,buffer_free);
	assert(read>=24);
	width = get_ushort(buffer_pos);
	height = get_ushort(buffer_pos+2);
	pixel_format.bpp = buffer_pos[4];
	pixel_format.depth = buffer_pos[5];
	for (yuri::ushort_t i = 0; i < 3; ++i) {
		pixel_format.colors[i].max = get_ushort(buffer_pos+8+i*2);
		pixel_format.colors[i].shift = *(buffer_pos+14+i);
	}
	yuri::uint_t name_len = get_uint(buffer_pos+20);
	std::string name(reinterpret_cast<const char*>(buffer_pos+24),name_len);
	log[info] << "handshake finished, connected to server " << name <<
			", with resolution " << width << "x" << height << std::endl;
	log[info] << "Server encoding uses " << pixel_format.bpp <<
			" bit per pixel, with " << pixel_format.depth <<
			" valid bits" << std::endl;
	log[info] << "Color parameters: " <<
			"red @" << static_cast<yuri::ushort_t>(pixel_format.colors[0].shift) << ", max: " <<  pixel_format.colors[0].max <<
			"green @" << static_cast<yuri::ushort_t>(pixel_format.colors[1].shift) << ", max: " <<  pixel_format.colors[1].max <<
			"blue @" << static_cast<yuri::ushort_t>(pixel_format.colors[2].shift) << ", max: " <<  pixel_format.colors[2].max << std::endl;
	image.reset(new yuri::ubyte_t[width*height*3]);
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
 *  \todo The method should throw an exception when unknows message type arrives.
 *
 *  \return true if some event was processed, false if more data are needed.
 */
bool VNCClient::process_data()
{
	if (!buffer_valid) return false;
	//yuri::ushort_t rectangles;
	std::string text;
	yuri::uint_t len;
	switch (state) {
		case awaiting_data:
			log[debug] << "Processing data, starting with " << static_cast<yuri::uint_t>(buffer_pos[0]) << ", valid: " << buffer_valid<< std::endl;
			switch(buffer_pos[0]) {
				case 0:if (buffer_valid < 4) return false;
					remaining_rectangles = get_ushort(buffer_pos+2);
					log[debug] << "Rectangles: " << remaining_rectangles<<std::endl;
					move_buffer(4);
					state = receiving_rectangles;
					//buffer_r
					return true;
				case 2: log[info] << "BELL" << std::endl;
					move_buffer(1);return true;
				case 3: if (buffer_valid < 8) return false;
					len = get_uint(buffer_pos+4);
					if (buffer_valid < 8+len) return false;
					text = std::string(reinterpret_cast<const char*>(buffer_pos+8),len);
					log[info] << "Server cut text: " << text << std::endl;
					move_buffer(8+len);
					return true;
				case 150: if (buffer_valid < 10) return false;
					log[info] << "Continuous updates enabled" << std::endl;
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
				yuri::ushort_t x,y,w,h;
				yuri::uint_t  enc;
				x = get_ushort(buffer_pos+0);
				y = get_ushort(buffer_pos+2);
				w = get_ushort(buffer_pos+4);
				h = get_ushort(buffer_pos+6);
				enc = get_uint(buffer_pos+8);
				log[debug] << "Got update for rect " << w<<"x"<<h<<" at " <<x<<"x"<<y<<", encoding: " << enc << std::endl;
				yuri::usize_t need = 0;
				switch (enc) {
					case 0: need = w*h*(pixel_format.bpp>>3);
						break;
					case 1: need = 4;
						log[warning] << "copy rect!" << std::endl;
						break;
					default:
						log[error] << "Unsupported encoding!!" << std::endl;
				}

				if (buffer_valid < 12 + need) {
					log[debug] << "Not enough data (need: " << need << ", have: " << buffer_valid << "@ "<< (buffer_pos - buffer.get())<< ")"<<std::endl;
					return false;
				}
				switch (enc) {
					case 0: {
						yuri::ubyte_t * pos = buffer_pos + 12, *outpos;
						yuri::size_t pixel;
						assert(pixel_format.bpp == 32);
						for (yuri::ushort_t line=y; line < y+h; ++line) {
							outpos = image.get()+3*width*line+x*3;
							for (yuri::ushort_t row = x; row < x + w; ++row) {
								pixel = get_pixel(pos);
								*outpos++ = (pixel>>pixel_format.colors[2].shift)&pixel_format.colors[2].max;
								*outpos++ = (pixel>>pixel_format.colors[1].shift)&pixel_format.colors[1].max;
								*outpos++ = (pixel>>pixel_format.colors[0].shift)&pixel_format.colors[0].max;
								pos += pixel_format.bpp >> 3;
							}
						}; }
						break;
					case 1: {
						yuri::ushort_t x0 = get_ushort(buffer_pos + 12);
						yuri::ushort_t y0 = get_ushort(buffer_pos + 14);
						yuri::ubyte_t *tmp, *pos=0;
						tmp = new yuri::ubyte_t[w*h*3];
						yuri::ubyte_t *b=tmp;
						for (yuri::ushort_t line=y0; line < y0+h; ++line) {
							pos = image.get()+3*width*line+x0*3;
							memcpy(b,pos,w*3);
							b+=w*3;
						}
						b = tmp;
						for (yuri::ushort_t line=y; line < y+h; ++line) {
							pos = image.get()+3*width*line+x*3;
							memcpy(pos,b,w*3);
							b+=w*3;
						}
						delete [] tmp;

						} break;

					default:
						log[error] << "Unimplemented encoding " << std::endl;
						break;
				}
				move_buffer(12+need);
				if (!--remaining_rectangles) {
					state = awaiting_data;
					shared_ptr<BasicFrame> frame = BasicIOThread::allocate_frame_from_memory(image,width*height*3);
					push_video_frame(0,frame,YURI_FMT_RGB24,width,height);
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
yuri::uint_t VNCClient::get_uint(yuri::ubyte_t *start)
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
yuri::ushort_t VNCClient::get_ushort(yuri::ubyte_t *start)
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
yuri::size_t VNCClient::get_pixel(yuri::ubyte_t *buf)
{
	//yuri::ushort_t b = 0;
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
void VNCClient::store_ushort(yuri::ubyte_t *start, yuri::ushort_t data)
{
	start[0]=(data>>8)&0xFF;
	start[1]=data&0xFF;
}
void VNCClient::store_uint(yuri::ubyte_t *start, yuri::uint_t data)
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
bool VNCClient::request_rect(yuri::ushort_t x, yuri::ushort_t y, yuri::ushort_t w, yuri::ushort_t h, bool incremental)
{
	yuri::ubyte_t b[10];
	b[0]=3;
	b[1]=static_cast<yuri::ubyte_t>(incremental);
	store_ushort(b+2,x);
	store_ushort(b+4,y);
	store_ushort(b+6,w);
	store_ushort(b+8,h);
	yuri::size_t size = socket->write(b,10);
	assert (size==10);
	return true;
}
/** \brief Sends a request for a continuous updates
 *  \return Always true
 */

bool VNCClient::enable_continuous()
{
	yuri::ubyte_t b[10];
	b[0]=150;
	b[1]=0;
	store_ushort(b+2,0);
	store_ushort(b+4,0);
	store_ushort(b+6,width);
	store_ushort(b+8,height);
	yuri::size_t size = socket->write(b,10);
	assert (size==10);
	return true;
}
bool VNCClient::set_encodings()
{
	yuri::ubyte_t b[12];
	b[0]=2;
	b[1]=0;
	store_ushort(b+2,2);
	store_uint(b+4,0);
	store_uint(b+8,1);
	yuri::size_t size = socket->write(b,12);
	assert (size==12);
	return true;
}

}

}
