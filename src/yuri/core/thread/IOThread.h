/*!
 * @file 		IOThread.h
 * @author 		Zdenek Travnicek
 * @date 		31.5.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */
#ifndef BASICIOTHREAD_H_
#define BASICIOTHREAD_H_

#ifdef YURI_LINUX
#include <sched.h>
#endif
#include "yuri/core/forward.h"
#include "yuri/core/utils/time_types.h"
#include "yuri/core/utils/Timer.h"
#include <vector>
#include <string>

#include "yuri/core/pipe/PipeNotification.h"
#include "yuri/core/thread/PipeConnector.h"
//#include "yuri/core/BasicIOMacros.h"
#include "yuri/core/thread/ThreadBase.h"

namespace yuri
{
namespace core
{

#define IOTHREAD_GENERATOR_DECLARATION 	static yuri::core::pIOThread generate(yuri::log::Log &log, yuri::core::pwThreadBase parent, const yuri::core::Parameters& parameters);
#define IOTHREAD_GENERATOR(cls) yuri::core::pIOThread cls::generate(yuri::log::Log &log,yuri::core::pwThreadBase parent, const yuri::core::Parameters& parameters)\
{ \
	try { \
		return make_shared<cls>(log,parent,parameters); \
	} \
	catch (std::exception &e) { \
		throw yuri::exception::InitializationFailed(std::string(#cls) + "constuctor failed: " + e.what()); \
	} \
}
#define IOTHREAD_INIT(parameters) \
		set_params(configure().merge(parameters));

class EXPORT IOThread: public ThreadBase, public PipeNotifiable
{
public:
	static Parameters			configure();
								IOThread(const log::Log &log_, pwThreadBase parent,
			position_t inp, position_t outp, const std::string& id = "IO");

	virtual 					~IOThread() noexcept;
/* ****************************************************************************
 * 							Input/Output
 **************************************************************************** */
	position_t	 				get_no_in_ports();
	position_t	 				get_no_out_ports();
	void 						connect_in(position_t index, pPipe pipe);
	void 						connect_out(position_t index, pPipe pipe);
//	void 				read_notification();
/* ****************************************************************************
 * 							Parameters
 **************************************************************************** */
//	virtual void 				set_affinity(yuri::ssize_t affinity);
//	virtual bool 				set_params(Parameters &parameters);
	virtual bool 				set_param(const Parameter &parameter) override;
//	template<typename T> bool 	set_param(const std::string& name, T value);
/* ****************************************************************************
 * 							Data allocation
 **************************************************************************** */

//	static pBasicFrame 			allocate_frame_from_memory(const yuri::ubyte_t *mem,
//			yuri::size_t size, bool large=false);
//	static pBasicFrame 			allocate_frame_from_memory(const plane_t& mem);
//
//	static pBasicFrame 			duplicate_frame(pBasicFrame frame);
//	static pBasicFrame 			allocate_empty_frame(yuri::format_t format,
//			yuri::size_t width, yuri::size_t height, bool large=false);
//	static pBasicFrame 			allocate_empty_frame(size_t size, bool large=false);
//	static bool 				connect_threads(pIOThread,
//			yuri::sint_t, pIOThread, yuri::sint_t, log::Log &log,
//			std::string name, pParameters params = pParameters());
//


/* ****************************************************************************
 * 							Protected API
 **************************************************************************** */
protected:
	virtual void 				run() override;
	virtual bool 				step();

	bool		 				push_frame(position_t index, pFrame frame);
	pFrame						pop_frame(position_t index);
	virtual void 				resize(position_t inp, position_t outp);
	position_t					do_get_no_in_ports();
	position_t					do_get_no_out_ports();
	void 						close_pipes();
	void 						set_latency(duration_t lat) { latency_=lat; }
	duration_t					get_latency() { return latency_; }
	bool 						pipes_data_available();
	virtual	void				do_connect_in(position_t, pPipe pipe);
	virtual	void				do_connect_out(position_t, pPipe pipe);

private:

	position_t 					in_ports_;
	position_t					out_ports_;
	mutex 						port_lock_;
	std::vector<PipeConnector > in_;
	std::vector<PipeConnector > out_;

	duration_t 					latency_;
	std::atomic<size_t>			active_pipes_;

	yuri::size_t 				fps_stats_;
	std::vector<yuri::size_t> 	streamed_frames_;
	std::vector<timestamp_t>	first_frame_;
	Timer 						pts_timer_;
};



}
}
#endif /*BASICIOTHREAD_H_*/

