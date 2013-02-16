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

#include "yuri/threads/ThreadBase.h"
#include "yuri/io/BasicPipe.h"
#include <vector>
#include <string>
#ifdef __linux__
#include <poll.h>
#endif
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include "yuri/io/PipeConnector.h"
#include "yuri/config/Config.h"
#include "yuri/config/Parameters.h"
#include "yuri/exception/NotImplemented.h"
#include "yuri/io/BasicIOMacros.h"

namespace yuri
{
namespace io
{
using yuri::threads::ThreadBase;
using yuri::threads::pThreadBase;
using yuri::log::Log;
using boost::mutex;
using namespace yuri::log;
class EXPORT BasicIOThread: public ThreadBase
{
public:
	static shared_ptr<Parameters> configure();
	static bool configure_converter(Parameters&, yuri::format_t ,yuri::format_t) throw(Exception);

	BasicIOThread(Log &log_, pThreadBase parent, yuri::sint_t inp, yuri::sint_t outp, std::string id = "IO");

	virtual ~BasicIOThread();
/* ****************************************************************************
 * 							Input/Output
 **************************************************************************** */
	virtual inline yuri::sint_t get_no_in_ports()
		{ boost::mutex::scoped_lock l(port_lock); return in_ports; }
	virtual inline yuri::sint_t get_no_out_ports()
		{ boost::mutex::scoped_lock l(port_lock); return out_ports; }
	virtual void connect_in(yuri::sint_t index,shared_ptr<BasicPipe> pipe);
	virtual void connect_out(yuri::sint_t index,shared_ptr<BasicPipe> pipe);
	virtual void close_pipes();
	virtual void set_latency(yuri::size_t lat) { latency=lat; }
	virtual bool pipes_data_available();
	virtual void read_notification();
/* ****************************************************************************
 * 							Parameters
 **************************************************************************** */
	virtual void set_affinity(yuri::ssize_t affinity);
	virtual bool set_params(Parameters &parameters);
	virtual bool set_param(Parameter &parameter);
	template<typename T> bool set_param(std::string name, T value);
/* ****************************************************************************
 * 							Data allocation
 **************************************************************************** */
//	static BasicFrame::plane_t allocate_memory_block(yuri::size_t size, bool large=false);
	static pBasicFrame allocate_frame_from_memory(const yuri::ubyte_t *mem, yuri::size_t size, bool large=false);
	static pBasicFrame allocate_frame_from_memory(const plane_t& mem);
//	static pBasicFrame allocate_frame_from_memory(shared_array<yuri::ubyte_t> mem, yuri::size_t size, bool large=false);
	static pBasicFrame duplicate_frame(pBasicFrame frame);
	static pBasicFrame allocate_empty_frame(yuri::format_t format, yuri::size_t width, yuri::size_t height, bool large=false);
	static bool connect_threads(shared_ptr<BasicIOThread>, yuri::sint_t, shared_ptr<BasicIOThread>, yuri::sint_t, Log &log,std::string name, shared_ptr<Parameters> params = shared_ptr<Parameters>());

protected:
	virtual void run();
	virtual bool step();
	virtual int set_fds();
	virtual void request_notifications();

	virtual bool push_raw_frame(yuri::sint_t index, pBasicFrame frame);
	virtual bool push_raw_video_frame(yuri::sint_t  index, pBasicFrame frame);
	virtual bool push_raw_audio_frame(yuri::sint_t  index, pBasicFrame frame);

	// Convenience functions for pushing frames
	virtual bool push_video_frame (yuri::sint_t index, pBasicFrame frame, yuri::format_t format, yuri::size_t width, yuri::size_t height, yuri::size_t pts, size_t duration, size_t dts);
	virtual bool push_video_frame (yuri::sint_t index, pBasicFrame frame, yuri::format_t format, yuri::size_t width, yuri::size_t height);
	virtual bool push_audio_frame (yuri::sint_t index, pBasicFrame frame, yuri::format_t format, yuri::usize_t channels, yuri::usize_t samples, yuri::size_t pts, yuri::size_t duration, yuri::size_t dts);
	virtual pBasicFrame timestamp_frame(pBasicFrame frame);
	virtual pBasicFrame get_frame_as(yuri::sint_t index, yuri::format_t format) DEPRECATED;

	Parameters params;
	std::vector<PipeConnector > in,out;
	yuri::sint_t in_ports,out_ports;
	virtual void resize(yuri::sint_t inp, yuri::sint_t outp);
	yuri::usize_t latency; // Latency in microseconds
#ifdef YURI_LINUX
	shared_array<struct pollfd> pipe_fds;
#endif
	yuri::uint_t active_pipes;
	mutex port_lock;
	yuri::ssize_t cpu_affinity;
	yuri::size_t fps_stats;
	std::vector<yuri::size_t> streamed_frames;
	std::vector<boost::posix_time::ptime> first_frame;
	boost::posix_time::ptime pts_base;
};

template<typename T> bool BasicIOThread::set_param(std::string name, T value)
{
	Parameter p(name,value);
	return set_param(p);
}


}
}
#endif /*BASICIOTHREAD_H_*/

