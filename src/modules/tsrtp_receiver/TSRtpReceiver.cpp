/*
 * TSRtpReceiver.cpp
 *
 *  Created on: Feb 22, 2011
 *      Author: worker
 */

#include "TSRtpReceiver.h"

namespace yuri {

namespace io {

REGISTER("ts_receiver",TSRtpReceiver)

shared_ptr<BasicIOThread> TSRtpReceiver::generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception)
{
	shared_ptr<TSRtpReceiver> yc (new TSRtpReceiver(_log,parent, parameters));
//yc->set_endpoint(parameters["address"].get<std::string>(),parameters["port"].get<yuri::size_t>());
	return yc;
}

shared_ptr<Parameters> TSRtpReceiver::configure()
{
	shared_ptr<Parameters> p (new Parameters());
	p->set_max_pipes(0,1);
	p->add_output_format(YURI_VIDEO_MPEGTS);
	// Just for now, let's define target as parameter
	(*p)["address"]["Adress to receive from"]=std::string();
	(*p)["port"]["Port to receive from"]="1234";
	(*p)["pass"]["Pass unprocessed RTP packets through"]=false;
	return p;
}

TSRtpReceiver::TSRtpReceiver(Log &log_, pThreadBase parent, Parameters &parameters):
				BasicIOThread(log_,parent,0,1,"TS RTP Streamer"),seq(0),pseq(0),
				buffer_size(1048576),buffer_position(0),pass_thru(false)
{
	params.merge(*configure());
	params.merge(parameters);

	pass_thru=parameters["pass"].get<bool>();
	buffer = allocate_memory_block(buffer_size);
	in_buffer = allocate_memory_block(buffer_size);

}

TSRtpReceiver::~TSRtpReceiver() {

}


bool TSRtpReceiver::set_endpoint(std::string address, yuri::size_t port)
{
	if (!socket) socket.reset(new ASIOUDPSocket(log,get_this_ptr(),port));
	return socket->set_endpoint(address,port);
}

void TSRtpReceiver::run()
{
	print_id(info);
	set_endpoint(params["address"].get<std::string>(),params["port"].get<yuri::size_t>());
	yuri::size_t read_len;
	while (still_running()) {
		read_len = socket->read(in_buffer.get(),buffer_size);
		log[verbose_debug] << "Read " << read_len << std::endl;
		if (pass_thru) {
			shared_ptr<BasicFrame> f = allocate_frame_from_memory(in_buffer.get(),read_len);
			push_raw_frame(0,f);
		} else {
			// The code bellow usually expects TS packets to be aligned to the beginning of the frame.
			// So we have to find sync byte and skip all preceding data
			if (read_len <= sizeof(RTPPacket)) continue;
			unsigned int pos = 0, len=0, total_len=buffer_position+read_len-sizeof(RTPPacket);
			//RTPPacket *packet = reinterpret_cast<RTPPacket*>(in_buffer.get());
			memcpy(buffer.get()+buffer_position,in_buffer.get()+sizeof(RTPPacket),read_len-sizeof(RTPPacket));
			while (pos < total_len) {
				if (buffer.get()[pos] == 0x47) {
					if (pos+2*188 >= total_len) {
						// We have data for only one more packet. let's waint till next packet
						buffer_position=total_len;
						pos = total_len;
						break;
					}
					if (buffer.get()[pos+188] == 0x47) break; // Second sync byte - we probably have the right offset
				}
				if (pos>total_len-188) {
					// No luck here ;)
					buffer_position = total_len;
					pos=total_len;
					break;
				}
				pos++;
			}
			if (pos+188>=total_len)  {
				//log[debug] << "Not enough data, waiting" << std::endl;
				continue;
			}
			//log[debug] << "pos: " << pos << " total: " << total_len << std::endl;
			len = total_len - pos;
			if (len % 188) len -= len % 188;
			shared_ptr<BasicFrame> f = allocate_frame_from_memory(buffer.get()+pos,len);
			//f->set_time(static_cast<yuri::size_t>(packet->timestamp));
			push_raw_frame(0,f);
			if (pos) log[debug] << "Threw away " << (pos) << " bytes out of " << total_len << std::endl;
			buffer_position=total_len-len-pos;
			if (buffer_position) {
				memmove(buffer.get(),buffer.get()+pos+len,buffer_position);
			}
		}
	}
}

}

}
