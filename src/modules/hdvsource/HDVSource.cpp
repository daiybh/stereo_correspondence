/*!
 * @file 		HDVSource.cpp
 * @author 		Zdenek Travnicek
 * @date 		29.5.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "HDVSource.h"
#include "yuri/core/Module.h"
namespace yuri {

namespace ieee1394 {

REGISTER("hdvsource",HDVSource)

core::pBasicIOThread HDVSource::generate(log::Log &_log,core::pwThreadBase parent, core::Parameters& parameters)
{
	shared_ptr<HDVSource> hdv(new HDVSource(_log,parent,
			parameters["node"].get<unsigned>(),
			parameters["port"].get<int>(),
			parameters["guid"].get<int64_t>()));
	return hdv;
}
core::pParameters HDVSource::configure()
{
	core::pParameters p (new core::Parameters());
	(*p)["node"]=0;
	(*p)["port"]=0;
	(*p)["guid"]=0;
	return p;
}


HDVSource::HDVSource(log::Log &log_,core::pwThreadBase parent, nodeid_t node, int port,
		int64_t guid): IEEE1394SourceBase(log_,parent,node,port,guid),
		total_packets(0),total_missing(0),buffer_size(0),buffer_position(0),
		enable_checks(true)
{
	log.set_label("[HDV Source] ");
	// Default size. This is proved to work correctly and it's just to fit single RTP packet.
	// User may change it any value he/she likes.
	setOutputBufferSize(1320);
}

HDVSource::~HDVSource() {
	log[log::debug] << "Received " << total_packets << " packets, " <<
		total_missing << " packet was missing in the stream" << std::endl;
	std::pair<int,int> pid;
	BOOST_FOREACH(pid,counters) {
		log[log::debug] << "There was pid " << pid.first << " in the received stream" << std::endl;
	}
}

bool HDVSource::start_receiving()
{
	mpeg_frame = iec61883_mpeg2_recv_init (handle,
			receive_frame, (void *)this);
	if (!mpeg_frame) return false;
	log[log::info] << "Calling receive start" << std::endl;
	if (iec61883_mpeg2_recv_start (mpeg_frame, channel))
		return false;
	log[log::info] << "Receiving" << std::endl;
	return true;
}

bool HDVSource::stop_receiving()
{
	if (mpeg_frame) iec61883_mpeg2_close (mpeg_frame);
	return true;
}

int HDVSource::receive_frame(unsigned char*data, int length, unsigned int dropped, void *source)
{
	return ((HDVSource*)source)->process_frame(data,length,dropped);
}

int HDVSource::process_frame(unsigned char *data, int length, unsigned int dropped)
{
	if (dropped) log[log::info] << "Dropped " << dropped << " frames" << std::endl;
	if (enable_checks) {
		int pid = static_cast<int>((static_cast<int>(data[1]&0x1F)*256 + data[2]));
		bool payload = static_cast<bool>(data[3]&0x10);
		bool error = static_cast<bool>(data[1]&0x80);
		if (error) {
			log[log::warning] << "Receiver error in stream, skipping" << std::endl;
			return 0;
		}
		log[log::verbose_debug] << "Received  frame with size " << length << " B from PID " <<
			pid << std::endl;
		if (*data!=0x47) log[log::warning] << "Missing sync byte!" << std::endl;
		int act_count = data[3]&0xF;
		/*log[log::debug] << "Packet header: 0x" << std::hex
			<< (int)(data[0]) << " " << (int)(data[1]) << " "
			<< (int)(data[2]) << " " << (int)(data[3]) <<
			std::dec << std::endl;*/
		if (payload) {
			if (act_count != counters[pid]) {
				log[log::warning] << "Continuity problem, expected " << counters[pid] <<
				", received " << act_count << ". PID : " << pid << std::endl;
	//			return 0;
				counters[pid]=(act_count+1)&0xF;
				//total_missing+=(act_count>counter)?act_count-counter:0xF+act_count-counter;
				total_missing++;
			} else counters[pid] = (counters[pid] + 1) & 0xF;
		}
	}

	//log[log::debug] << "Counter: " << (int)(data[3]&0xF) << std::endl;
	total_packets++;
	if (out[0]) {
		boost::mutex::scoped_lock l(buffer_lock);
		if (!output_buffer.size()) {
			//l.unlock();
			//out[0]->push_frame(data,length);
			do_send_data(reinterpret_cast<yuri::ubyte_t*>(data),length);
		} else {
			if (buffer_position + length > buffer_size) do_sendOutputBuffer();
			if (length > buffer_size) {
				//l.unlock();
				//out[0]->push_frame(data,length);
				do_send_data(reinterpret_cast<yuri::ubyte_t*>(data),length);
			} else {
				//memcpy(output_buffer.get() + buffer_position,data,length);
				std::copy(data,data+length,output_buffer.begin()+buffer_position);
				buffer_position += length;
			}
		}
	}
	else {
		log[log::warning] << "Received frame and don't have pipe to put it into, throwing away" << std::endl;
	}
	return 0;
}

void HDVSource::setOutputBufferSize(long size)
{
	boost::mutex::scoped_lock l(buffer_lock);
	if (size == buffer_size) return;
	do_sendOutputBuffer();
	buffer_size = size;
	if (size) output_buffer.resize(size);
	else output_buffer.resize(0);

}

void HDVSource::do_sendOutputBuffer()
{
	if (buffer_position && output_buffer.size())
	do_send_data(&output_buffer[0],buffer_position);
	buffer_position = 0;
}

void HDVSource::do_send_data(yuri::ubyte_t *data, yuri::size_t size)
{
	core::pBasicFrame frame = allocate_frame_from_memory(data,size);
	push_video_frame(0,frame,YURI_VIDEO_MPEGTS,1440,1080);
}

}


}
