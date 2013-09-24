/*!
 * @file 		IOThread.cpp
 * @author 		Zdenek Travnicek
 * @date 		31.5.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "IOThread.h"
#include "yuri/exception/NotImplemented.h"
#include "yuri/core/frame/Frame.h"
#include "yuri/core/pipe/Pipe.h"
#include <algorithm>
#include <stdexcept>

namespace yuri
{
namespace core
{

Parameters IOThread::configure()
{
	Parameters p = ThreadBase::configure();//;// = make_shared<Parameters>();
//	p["cpu"]["Bind thread to cpu"]=-1;
	p["fps_stats"]["Print out_ current FPS every n frames. Set to 0 to disable."]=0;
//	p["debug"]["Change debug level. value 0 will keep inherited value from app, lower numbers will reduce verbosity, higher numbers will make output more verbose."]=0;
//	p["node_name"]["Name of the node. Will be filled automatically by the builder."]="";
	return p;
}



//pBasicFrame IOThread::allocate_frame_from_memory(const yuri::ubyte_t *mem, yuri::size_t size, bool large)
//{
//	pBasicFrame f = make_shared<BasicFrame>(1);
//	if (!large) {
//		f->set_plane(0,mem,size);
//	} else {
//		FixedMemoryAllocator::memory_block_t block = FixedMemoryAllocator::get_block(size);
//		assert(block.first);
//		std::copy(mem,mem+size,&block.first[0]);
//		f->get_plane(0).set(block.first,size,block.second);
//	}
//	return f;
//}
//pBasicFrame IOThread::allocate_frame_from_memory(const plane_t& mem)
//{
//	pBasicFrame f = make_shared<BasicFrame>(1);
//	f->set_plane(0,mem);
//	return f;
//}
//
//pBasicFrame IOThread::duplicate_frame(pBasicFrame frame)
//{
//	pBasicFrame f = frame->get_copy();
//	return f;
//}

IOThread::IOThread(const log::Log &log_,pwThreadBase parent, position_t inp, position_t outp, const std::string& id):
	ThreadBase(log_, parent, id), in_ports_(inp), out_ports_(outp),latency_(200_ms),
	active_pipes_(0),
	fps_stats_(0)

{
	TRACE_METHOD
	//params.merge(*configure());
	log.set_label(get_node_name());
	resize(inp,outp);
}

IOThread::~IOThread() noexcept
{
	TRACE_METHOD
	close_pipes();
}

void IOThread::run()
{
	TRACE_METHOD
//	IO_THREAD_PRE_RUN
	try {
		while (still_running()) {
			if (!active_pipes_ && in_ports_ ) {
				ThreadBase::sleep(latency_);
			}
			if (in_ports_ && !pipes_data_available()) {
				wait_for(latency_);
			}
			log[log::verbose_debug] << "Stepping" << "\n";
			if (!step()) break;
		}
	}
	catch (std::runtime_error& e) {
		log[log::debug] << "Thread failed: " << e.what();
	}

	close_pipes();
//	IO_THREAD_POST_RUN
}
position_t IOThread::get_no_in_ports()
{
	lock_t _(port_lock_);
	return do_get_no_in_ports();
}
position_t IOThread::do_get_no_in_ports()
{
	return in_ports_;
}
position_t IOThread::get_no_out_ports()
{
	lock_t _(port_lock_);
	return do_get_no_out_ports();
}
position_t IOThread::do_get_no_out_ports()
{
	return out_ports_;
}

void IOThread::connect_in(position_t index, pPipe pipe)
{
	TRACE_METHOD
	lock_t _(port_lock_);
	do_connect_in(index, pipe);
}
void IOThread::do_connect_in(position_t index, pPipe pipe)
{
	TRACE_METHOD
	if (index < 0 || index >= do_get_no_in_ports()) throw std::out_of_range("Input pipe out of Range");
	if (in_[index]) {
		log[log::debug] << "Disconnecting already connected pipe from in port " << index << "\n";
	}
	in_[index]=PipeConnector(pipe,dynamic_pointer_cast<PipeNotifiable>(get_this_ptr()));
	active_pipes_ = std::accumulate(in_.begin(), in_.end(), 0, [](const size_t& ap, const PipeConnector&p){return ap + (p?1:0);});
//	set_fds();

}
void IOThread::connect_out(position_t index, pPipe pipe)
{
	TRACE_METHOD
	do_connect_out(index, pipe);
}
void IOThread::do_connect_out(position_t index, pPipe pipe)
{
	TRACE_METHOD
	if (index < 0 || index >= do_get_no_out_ports()) throw std::out_of_range("Output pipe out of Range");
	if (out_[index]) log[log::debug] << "Disconnecting already connected pipe from out port " << index << "\n";
	out_[index]=PipeConnector(pipe,dynamic_pointer_cast<PipeNotifiable>(get_this_ptr()));
}
bool IOThread::push_frame(position_t index, pFrame frame)
{
	TRACE_METHOD
	return out_[index]->push_frame(frame);
}

pFrame	IOThread::pop_frame(position_t index)
{
	TRACE_METHOD
	return in_[index]->pop_frame();
}

void IOThread::resize(position_t inp, position_t outp)
{
	TRACE_METHOD
	log[log::debug] << "Resizing to " << inp << " input ports and " << outp << " output ports." << "\n";
	if (inp >= 0) in_ports_	= inp;
	if (outp>= 0) out_ports_= outp;
//	PipeConnector pc(dynamic_pointer_cast<PipeNotifiable>(get_this_ptr()));
	in_.resize(in_ports_);
	out_.resize(out_ports_);
	streamed_frames_.resize(out_ports_,0);
	first_frame_.resize(out_ports_);
}

void IOThread::close_pipes()
{
	TRACE_METHOD
	log[log::debug] << "Closing pipes!";
	for (auto& pipe: out_) {
		if (pipe) pipe->close_pipe();
		pipe.reset();
	}
}

bool IOThread::pipes_data_available()
{
	TRACE_METHOD
	for (auto& pipe: in_) {
		if (!pipe) continue;
		if (!pipe->is_empty()) return true;
		if (pipe->is_finished()) {
			pipe.reset();
			active_pipes_--;
		}
	}
	return false;
}

/*pBasicFrame IOThread::timestamp_frame(pBasicFrame frame)
{
#ifndef YURI_USE_CXX11
	if (pts_base==boost::posix_time::not_a_date_time) pts_base=boost::posix_time::microsec_clock::local_time();
	boost::posix_time::time_duration pts = boost::posix_time::microsec_clock::local_time() - pts_base;
	frame->set_time(pts.total_microseconds(),frame->get_dts(),frame->get_duration());
#else
	if (pts_base.time_since_epoch() == time_duration::zero()) {
		pts_base = std::chrono::steady_clock::now();
	}
	time_duration pts = std::chrono::steady_clock::now() - pts_base;
	frame->set_time(std::chrono::duration_cast<std::chrono::microseconds>(pts).count(),frame->get_dts(),frame->get_duration());
#endif
	return frame;
}
// TODO Stub, not implemented!!!!!!
//pBasicFrame IOThread::get_frame_as(yuri::sint_t index, yuri::format_t format)
//{
//	pBasicFrame frame;
//	if (index>=in_ports_ || !in_[index] || in_[index]->is_empty()) return frame;
//	if (format == YURI_FMT_NONE) return in_[index]->pop_frame();
//
//	return pBasicFrame();
//}
 * */

//void IOThread::set_affinity(yuri::ssize_t affinity)
//{
//	cpu_affinity = affinity;
//}
 /*
bool IOThread::set_params(Parameters &parameters)
{
#ifndef YURI_USE_CXX11
	std::pair<std::string,shared_ptr<Parameter> > par;
	BOOST_FOREACH(par,(parameters.params)) {
#else
	for(auto& par: parameters.params) {
#endif
		if (par.second && !set_param(*(par.second))) return false;
	}
	return true;
}*/
/* Sets one parameter.
 *
 * @param:  parameter - parameter to set
 * @return false on error (like value can't be set) and true otherwise (even for unknown parameter)
 *
 */
bool IOThread::set_param(const Parameter &parameter)
{
	if (parameter.get_name() == "fps_stats") {
		fps_stats_ = parameter.get<yuri::size_t>();
	} else return ThreadBase::set_param(parameter);
	return true;
}

/*
pBasicFrame IOThread::allocate_empty_frame(yuri::format_t format, yuri::size_t width, yuri::size_t height, bool large)
{
	pBasicFrame pic;
	FormatInfo_t fmt = BasicPipe::get_format_info(format);
	assert(fmt->planes && !fmt->compressed); // Just to make sure the format is valid
	pic.reset(new BasicFrame(fmt->planes));
	yuri::size_t bpplane = fmt->bpp;
	for (yuri::size_t i = 0; i < fmt->planes; ++i) {
		if (fmt->planes>1) bpplane = fmt->component_depths[i];
		yuri::size_t planesize = width * height * bpplane / fmt->plane_x_subs[i] / fmt->plane_y_subs[i] / 8;
		if (!large) {
			pic->get_plane(i).resize(planesize);
		} else {
			FixedMemoryAllocator::memory_block_t block = FixedMemoryAllocator::get_block(planesize);
			assert(block.first);
			//std::copy(mem,mem+size,&block.first[0]);
			pic->get_plane(i).set(block.first,planesize,block.second);
		}
		pic->get_plane(i).resize(planesize);
	}
	pic->set_parameters(format,width,height);
	return pic;
}
pBasicFrame IOThread::allocate_empty_frame(size_t size, bool large)
{
	pBasicFrame frame = yuri::make_shared<BasicFrame>(1);
	if (!large) {
		frame->get_plane(0).resize(size);
	} else {
		FixedMemoryAllocator::memory_block_t block = FixedMemoryAllocator::get_block(size);
		assert(block.first);
		frame->get_plane(0).set(block.first,size,block.second);
	}
	return frame;
}

bool IOThread::connect_threads(shared_ptr<BasicIOThread> src, yuri::sint_t s_idx, shared_ptr<BasicIOThread> target, yuri::sint_t t_idx, Log &log,std::string name, shared_ptr<Parameters> params)
{
	shared_ptr<Parameters> p = BasicPipe::configure();
	if (params) p->merge(*params);
	shared_ptr<BasicPipe> pipe = BasicPipe::generator(log,name,*p);
	target->connect_in(s_idx,pipe);
	src->connect_out(t_idx,pipe);
	return true;
}

*/


}
}

// End of File