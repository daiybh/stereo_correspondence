/*!
 * @file 		BasicIOThread.h
 * @author 		Zdenek Travnicek
 * @date 		31.5.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */
#ifndef BASICIOTHREAD_H_
#define BASICIOTHREAD_H_

#ifdef __linux__
#include <sched.h>
#endif
#include "yuri/core/forward.h"
#include <vector>
#include <string>
#ifdef __linux__
#include <poll.h>
#endif
#include "yuri/core/PipeConnector.h"
#include "yuri/core/Parameters.h"
#include "yuri/core/BasicIOMacros.h"
#include "yuri/core/ThreadBase.h"

namespace yuri
{
namespace core
{

class EXPORT BasicIOThread: public ThreadBase
{
public:
	static pParameters			configure();
	static bool 				configure_converter(Parameters&, yuri::format_t ,
				yuri::format_t);

								BasicIOThread(log::Log &log_, pwThreadBase parent,
			yuri::sint_t inp, yuri::sint_t outp, std::string id = "IO");

	virtual 					~BasicIOThread();
/* ****************************************************************************
 * 							Input/Output
 **************************************************************************** */
	virtual inline yuri::sint_t get_no_in_ports()
		{ boost::mutex::scoped_lock l(port_lock); return in_ports; }
	virtual inline yuri::sint_t get_no_out_ports()
		{ boost::mutex::scoped_lock l(port_lock); return out_ports; }
	virtual void 				connect_in(yuri::sint_t index, pBasicPipe pipe);
	virtual void 				connect_out(yuri::sint_t index, pBasicPipe pipe);
	virtual void 				close_pipes();
	virtual void 				set_latency(yuri::size_t lat) { latency=lat; }
	virtual bool 				pipes_data_available();
	virtual void 				read_notification();
/* ****************************************************************************
 * 							Parameters
 **************************************************************************** */
	virtual void 				set_affinity(yuri::ssize_t affinity);
	virtual bool 				set_params(Parameters &parameters);
	virtual bool 				set_param(const Parameter &parameter);
	template<typename T> bool 	set_param(std::string name, T value);
/* ****************************************************************************
 * 							Data allocation
 **************************************************************************** */
//	static BasicFrame::plane_t allocate_memory_block(yuri::size_t size, bool large=false);
	static pBasicFrame 			allocate_frame_from_memory(const yuri::ubyte_t *mem,
			yuri::size_t size, bool large=false);
	static pBasicFrame 			allocate_frame_from_memory(const plane_t& mem);
//	static pBasicFrame allocate_frame_from_memory(shared_array<yuri::ubyte_t> mem, yuri::size_t size, bool large=false);
	static pBasicFrame 			duplicate_frame(pBasicFrame frame);
	static pBasicFrame 			allocate_empty_frame(yuri::format_t format,
			yuri::size_t width, yuri::size_t height, bool large=false);
	static pBasicFrame 			allocate_empty_frame(size_t size, bool large=false);
	static bool 				connect_threads(pBasicIOThread,
			yuri::sint_t, pBasicIOThread, yuri::sint_t, log::Log &log,
			std::string name, pParameters params = pParameters());

protected:
	virtual void 				run();
	virtual bool 				step();
	virtual int 				set_fds();
	virtual void 				request_notifications();

	virtual bool 				push_raw_frame(yuri::sint_t index, pBasicFrame frame);
	virtual bool 				push_raw_video_frame(yuri::sint_t  index, pBasicFrame frame);
	virtual bool 				push_raw_audio_frame(yuri::sint_t  index, pBasicFrame frame);

	// Convenience functions for pushing frames
	virtual bool 				push_video_frame (yuri::sint_t index, pBasicFrame frame,
			yuri::format_t format, yuri::size_t width, yuri::size_t height,
			yuri::size_t pts, size_t duration, size_t dts);
	virtual bool 				push_video_frame (yuri::sint_t index, pBasicFrame frame,
			yuri::format_t format, yuri::size_t width, yuri::size_t height);
	virtual bool 				push_audio_frame (yuri::sint_t index, pBasicFrame frame,
			yuri::format_t format, yuri::usize_t channels, yuri::usize_t samples,
			yuri::size_t pts, yuri::size_t duration, yuri::size_t dts);
	virtual pBasicFrame 		timestamp_frame(pBasicFrame frame);
//	virtual pBasicFrame 		get_frame_as(yuri::sint_t index, yuri::format_t format) DEPRECATED;
	virtual void				set_log_id();
	Parameters 					params;
	std::vector<PipeConnector > in;
	std::vector<PipeConnector > out;
	yuri::sint_t 				in_ports;
	yuri::sint_t				out_ports;
	virtual void 				resize(yuri::sint_t inp, yuri::sint_t outp);
	yuri::usize_t 				latency; // Latency in microseconds
#ifdef YURI_LINUX
	shared_array<struct pollfd> pipe_fds;
#endif
	yuri::uint_t 				active_pipes;
	mutex 						port_lock;
	yuri::ssize_t 				cpu_affinity;
	yuri::size_t 				fps_stats;
	std::vector<yuri::size_t> 	streamed_frames;
	std::vector<boost::posix_time::ptime>
								first_frame;
	boost::posix_time::ptime 	pts_base;
	std::string 				node_id_;
	std::string					node_name_;
};

template<typename T> bool BasicIOThread::set_param(std::string name, T value)
{
	Parameter p(name,value);
	return set_param(p);
}


}
}
#endif /*BASICIOTHREAD_H_*/

