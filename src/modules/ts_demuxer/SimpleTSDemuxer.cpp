/*
 * SimpleTSDemuxer.cpp
 *
 *  Created on: Oct 3, 2010
 *      Author: neneko
 */

#include "SimpleTSDemuxer.h"
#include "yuri/core/Module.h"
namespace yuri {

namespace ts_demux {
REGISTER("simple_ts_demuxer",SimpleTSDemuxer)

IO_THREAD_GENERATOR(SimpleTSDemuxer)

core::pParameters SimpleTSDemuxer::configure()
{
	core::pParameters p = BasicIOThread::configure();
	(*p)["program"]["Program to demux"]=0;
	(*p)["buffer"]["Size of buffer"]=1048576;
	(*p)["parse_stream"]["Parse stream and look for resolution. No output until resolution is determined!"]=true;
	(*p)["width"]["Force output resolution. Ignored when parse_stream = true"]=720;
	(*p)["height"]["Force output resolution. Ignored when parse_stream = true"]=576;
	(*p)["packet_size"]["Size of packet. For normal stream this should be 188, for MTS (TS from camera) this should be 192"]=188;
	(*p)["strip"]["Strip PES headers"]=true;
	(*p)["quick_strip"]["Use faster stripping algorithm. Disable if the stripped output is garbled"]=true;
	p->set_max_pipes(1,1);
	p->add_input_format(YURI_VIDEO_MPEGTS);
	p->add_output_format(YURI_VIDEO_MPEG2);
	p->add_converter(YURI_VIDEO_MPEGTS,YURI_VIDEO_MPEG2,false);
	return p;
}

SimpleTSDemuxer::SimpleTSDemuxer(log::Log &log_, core::pwThreadBase parent, core::Parameters& parameters):
		BasicIOThread(log_,parent,1,1,"Simple TS Demux"),program(0),
		buffer_size(1048576),buffer_position(0),in_buffer_position(0),
		packet_size(188),parse_stream(false),width(-1),height(-1),pts(0)
		,last_pts(0)
{
	IO_THREAD_INIT("TS Demuxer")
	//in_buffer.reset(new yuri::ubyte_t[2*packet_size]);
	in_buffer.resize(2*packet_size);
}

SimpleTSDemuxer::~SimpleTSDemuxer() {
	std::pair<yuri::uint_t,yuri::size_t> pid;
	BOOST_FOREACH(pid,seen_pids) {
		log[log::info] << "Encountered total of " << pid.second <<
			" packets for PID " << pid.first << "\n";
	}
}

bool SimpleTSDemuxer::step()
{
	if (!in[0]) return true;
	core::pBasicFrame frame;
	while ((frame = in[0]->pop_frame())) {
		yuri::size_t remaining = PLANE_SIZE(frame,0)+in_buffer_position;
		yuri::size_t index = 0;
		pts = frame->get_pts();
		if (!last_pts) last_pts=pts;
		while (remaining >= packet_size) {
			yuri::size_t balast = packet_size-188;
			if (in_buffer_position) {
				yuri::ssize_t sync_offset=find_sync(reinterpret_cast<uint8_t*>(&in_buffer[0]+balast),in_buffer_position-balast);
				if (sync_offset<0) {
					in_buffer_position=0;
					continue;
				}

				if (sync_offset) {
					log[log::debug] << "Found sync " << sync_offset << " bytes after it's position!\n";
				}
				/*if (remaining-sync_offset < packet_size) {

				}*/
				yuri::size_t missing=packet_size-in_buffer_position+sync_offset-balast;
				memcpy(&in_buffer[0]+in_buffer_position,PLANE_RAW_DATA(frame,0)+sync_offset-balast,missing);
				process_packet(&in_buffer[0]+sync_offset);
				index+=missing; remaining-=missing+sync_offset;
				in_buffer_position = 0;
			} else {
				yuri::ssize_t sync_offset=find_sync(reinterpret_cast<uint8_t*>(PLANE_RAW_DATA(frame,0)+index+balast),remaining-balast);
				if (sync_offset<0) {
					in_buffer_position=0;
					log[log::debug] << "No sync in stream, skipping " << remaining << " bytes\n";
					remaining=0;
					continue;
				}
				if (sync_offset) {
					remaining-=sync_offset;
					index+=sync_offset;
					log[log::debug] << "Found sync " << sync_offset << " bytes after it's position!\n";
					continue;
				}
				process_packet(PLANE_RAW_DATA(frame,0)+index);
				index+=packet_size; remaining-=packet_size;
			}
		}
		if (remaining > 0) {
			memcpy(&in_buffer[0],PLANE_RAW_DATA(frame,0)+index,remaining);
			in_buffer_position = remaining;
		}
	}
	return true;
}

bool SimpleTSDemuxer::process_packet(yuri::ubyte_t *dataraw)
{
	if (packet_size < 188) {
		return false;
	}
	yuri::size_t offset = packet_size - 188;
	yuri::ubyte_t *data = dataraw+offset;
	yuri::uint_t pid = ((reinterpret_cast<unsigned char*>(data)[1]&0x1F) << 8)
				+ (reinterpret_cast<unsigned char*>(data)[2]&0xFF);
			//static_cast<uint>((static_cast<uint>(data[1]&0x1F)*256 + data[2]&0xFF));
	bool payload = static_cast<bool>(data[3]&0x10);
	bool adaptation = static_cast<bool>(data[3]&0x20);
	bool error = static_cast<bool>(data[1]&0x80);
	char sync = data[0];
	bool start = static_cast<bool>(data[1]&0x40);
	yuri::size_t skip_bytes = 4;
	yuri::size_t seq /*YURI_UNUSED */= static_cast<yuri::size_t>(data[3]&0xF);
	if (sync != 0x47) {
		log[log::debug] << "Sync not found! skipping frame\n";
		return true;
	}
	if (seen_pids.find(pid) == seen_pids.end()) {
		seen_pids[pid]=1;
		log[log::info] << "Found PID "<< pid << "\n";
	} else {
		seen_pids[pid]++;
	}
	//log[log::debug] << "PID: " << pid <<endl;
	if (error || !payload || pid != program) return true;
	if (adaptation) {
		skip_bytes += reinterpret_cast<uint8_t*>(data)[5]+1;
		log[log::info] << "Found adaptation field, skipping " << skip_bytes << "\n";
		}
	if (skip_bytes >= 188) return true;
	bool rai = false;
	//log[log::debug] << "Adaptation " << adaptation << endl;
	if (adaptation) {
		rai = static_cast<bool>(data[6]&0x40);
			//log[log::debug] << "Adaptation: rai: " << static_cast<bool>(data[6]&0x40)
				//<< ", start: " << start << endl;
	}
	if (start) {
		log[log::debug] << "pts: " << pts << ", start: " << start << "\n";
	}
	if (start && buffer_position) send_buffer();
	if (buffer_position + 188 - skip_bytes > buffer_size) send_buffer();
	memcpy(&buffer[0]+buffer_position,data+skip_bytes,188-skip_bytes);
	log[log::debug] << "Copying: " << 188-skip_bytes << " bytes (skipping " << skip_bytes<< ", bp: "<< buffer_position<<"\n";
	buffer_position+=188-skip_bytes;
	return true;
}

void SimpleTSDemuxer::set_buffer_size(yuri::size_t buf_size)
{
	buffer_size = buf_size;
	buffer_position = 0;
	buffer.resize(buffer_size);
}

void SimpleTSDemuxer::send_buffer()
{
	if (out[0] && buffer_position) {
		if (parse_stream && (width <0 || height < 0)) {
			process_headers();
		}
		if (width > 0 && height > 0) {
			core::pBasicFrame frame;
			if (strip_pes) {
				log[log::debug] << "Stripping pes\n";
				frame=stripped_pes();
			}
			else frame = allocate_frame_from_memory(&buffer[0],buffer_position);
			if (frame) {
				push_video_frame(0,frame,YURI_VIDEO_MPEG2,width,height);
				log[log::debug] << "Sent pts" << pts << "\n";
			}
		}
	} else {
		log[log::debug] << "Not sending!!\n";
	}
	buffer_position = 0;

}

void SimpleTSDemuxer::process_headers()
{
	unsigned char *buf = reinterpret_cast<unsigned char*>(&buffer[0]);
	unsigned char *end_buf=buf+buffer_position;
	while (buf<end_buf) {
		if (*(buf++) != 0) continue;
		if (*(buf++) != 0) continue;
		if (*(buf++) != 1) continue;
		unsigned short htype = *buf++;
		switch (htype) {
		case 0xb3:
			width = *buf++ << 4;
			width |= *buf>>4;
			height = *buf++ << 8 & 0xF00;
			height |= *buf++;

			std::string ar="undefined";
			switch (*buf >> 4) {
				case 1: ar = "1:1";break;
				case 2: ar = "4:3";break;
				case 3: ar = "16:9";break;
				case 4: ar = "2.21:1";break;
			}
			std::string fps = "undefined";
			switch (*buf &0xF) {
				case 1: fps = "23.976"; break;
				case 2: fps = "24"; break;
				case 3: fps = "25"; break;
				case 4: fps = "29.97"; break;
				case 5: fps = "30"; break;
				case 6: fps = "50"; break;
				case 7: fps = "59.94"; break;
				case 8: fps = "60"; break;
			}
			log[log::debug] << "Found MPEG seq header. w: " << width <<", h: "
					<< height << ", with aspect ratio: " << ar << " and " << fps
					<< " frames per second\n";
			break;
		/*case 0xb5:
			log[log::debug] << "Found MPEG extension header " << hex << htype << " ("<< static_cast<unsigned short>(*buf++>>4)<< ") "<<  dec << endl;
			break;
		default:
			log[log::debug] << "Found MPEG header " << hex << htype << dec << endl;*/
		}

	}
}

bool SimpleTSDemuxer::set_param(const core::Parameter &parameter)
{
	if (parameter.name == "program") {
		program=parameter.get<yuri::uint_t>();
	} else if (parameter.name == "parse_stream") {
		parse_stream=parameter.get<bool>();
		if (parse_stream) {
			width = -1;
			height = -1;
		}
	} else if (parameter.name == "width") {
		if (!parse_stream)
			width=parameter.get<int>();
	} else if (parameter.name == "height") {
		if (!parse_stream)
			height=parameter.get<int>();
	} else if (parameter.name == "buffer") {
		set_buffer_size(parameter.get<yuri::size_t>());
	} else if (parameter.name == "packet_size") {
		packet_size=parameter.get<yuri::size_t>();
		if (packet_size < 188) {
			log[log::error] << "Packet size has to be greater than 188 bytes! Setting to 188\n";
			packet_size = 188;
		}
		in_buffer.resize(2*packet_size);
	} else if (parameter.name == "strip") {
		strip_pes=parameter.get<bool>();
	} else if (parameter.name == "quick_strip") {
		quick_strip=parameter.get<bool>();
	}else return BasicIOThread::set_param(parameter);
	return true;
}

core::pBasicFrame SimpleTSDemuxer::stripped_pes()
{
	//shared_array<yuri::ubyte_t> out_buf = allocate_memory_block(buffer_position);
	plane_t out_buf(buffer_position);
	yuri::size_t position = 0, last_start=0, out_pos=0, header_length, data_length;
	yuri::size_t skipped=0, copied=0;
	bool null_size = false;
	uint8_t *buf = reinterpret_cast<uint8_t*>(&buffer[0]);
	while (buffer_position - position > 9) {
		if (buf[position] != 0 ||
			buf[position+1] != 0 ||
			buf[position+2] != 1 ||
			(buf[position+3] & 0xE0) != 0xE0) {
				position++;
				if (!null_size) skipped++;
				continue; // TO search through the stream
				//break; // TO have only the first header
		}
		if (null_size) {
			null_size = false;
			data_length = position-last_start;
			memcpy(&out_buf[0]+out_pos, buf+last_start,data_length);
			copied+=data_length;
			out_pos+=data_length;
		}
		data_length = (buf[4] << 8) + buf[5];
		header_length = buf[position + 8]+9;
		if (data_length) {
			if (position+header_length+data_length > buffer_position) {
				log[log::error] << "PES contains data beyond packet!\n";
			} else {
				memcpy(&out_buf[0]+out_pos, buf+position+header_length,data_length);
				out_pos+=data_length;
				copied+=data_length;
			}
			position+=header_length + data_length;
		} else {
			null_size = true;
			last_start = position+header_length;
			if (quick_strip) {
				position=buffer_position;
				break;
			}
			position+=header_length;
		}
	}
	if (null_size) {
		null_size = false;
		data_length = position-last_start;
		memcpy(&out_buf[0]+out_pos, buf+last_start,data_length);
		copied+=data_length;
		out_pos+=data_length;
	}
	/*position = buffer_position;
	if (last_start!=position) {
		memcpy(out_buf.get()+out_pos, buf+last_start,position-last_start);
		out_pos+=position-last_start;
	}*/
	log[log::debug] << "Copied " << copied << " bytes from " << buffer_position <<
			", skipped " << skipped << " bytes\n";
	if (!copied) return core::pBasicFrame();
	return allocate_frame_from_memory(&out_buf[0],out_pos);
}

yuri::ssize_t SimpleTSDemuxer::find_sync(uint8_t *data,yuri::size_t len)
{
	for (yuri::size_t i=0;i<len;++i) {
		if (data[i]==0x47) return i;
	}
	return -1;
}

}

}

