/*!
 * @file 		BasicIOThread.cpp
 * @author 		Zdenek Travnicek
 * @date 		31.5.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "BasicIOThread.h"
#include "yuri/core/FixedMemoryAllocator.h"
#include "yuri/exception/NotImplemented.h"
//#include <boost/make_shared.hpp>
#include "yuri/core/BasicFrame.h"
#include "yuri/core/BasicPipe.h"
#include <algorithm>
#include "boost/date_time/posix_time/posix_time.hpp"
using namespace yuri::log;

namespace yuri
{
namespace core
{

#ifdef BASICIOTHREAD_ENABLE_PORT_LOCK
#define DEBUG_LOCKS log[verbose_debug] << "Locking ports, " << __FILE__ << ":" << __LINE__ << " with lock @ " << (void*)(&port_lock) << "\n";
#endif

/*shared_ptr<BasicIOThread> BasicIOThread::generate(Log &_log,pThreadBase parent,
		Parameters& parameters) throw (Exception)
{
	throw (InitializationFailed("Not implemented!"));
}
*/
shared_ptr<Parameters> BasicIOThread::configure()
{
	shared_ptr<Parameters> p(new Parameters());
	(*p)["cpu"]["Bind thread to cpu"]=-1;
	(*p)["fps_stats"]["Print out current FPS every n frames. Set to 0 to disable."]=0;
	(*p)["debug"]["Change debug level. value 0 will keep inherited value from app, lower numbers will reduce verbosity, higher numbers will make output more verbose."]=0;
	return p;
}

bool BasicIOThread::configure_converter(Parameters&, yuri::format_t,yuri::format_t)
{
	throw exception::NotImplemented();
}

//shared_array<yuri::ubyte_t> BasicIOThread::allocate_memory_block(yuri::size_t size, bool large)
//{
//	shared_array<yuri::ubyte_t> mem;
//	if (!large) mem.reset(new yuri::ubyte_t[size]);
//	else mem = FixedMemoryAllocator::get_block(size);
//	return mem;
//}
pBasicFrame BasicIOThread::allocate_frame_from_memory(const yuri::ubyte_t *mem, yuri::size_t size, bool large)
{
//	shared_array<yuri::ubyte_t> smem = allocate_memory_block(size,large);
//	memcpy(smem.get(),mem,size);
//	return allocate_frame_from_memory(smem,size,large);
	pBasicFrame f = make_shared<BasicFrame>(1);
	//std::copy(mem,mem+size,f->get_plane(0).begin());
	if (!large) {
		f->set_plane(0,mem,size);
	} else {
		FixedMemoryAllocator::memory_block_t block = FixedMemoryAllocator::get_block(size);
		assert(block.first);
		std::copy(mem,mem+size,&block.first[0]);
		f->get_plane(0).set(block.first,size,block.second);
	}
	return f;
}
pBasicFrame BasicIOThread::allocate_frame_from_memory(const plane_t& mem)
{
	pBasicFrame f = make_shared<BasicFrame>(1);
	f->set_plane(0,mem);
	return f;
}

//pBasicFrame BasicIOThread::allocate_frame_from_memory(shared_array<yuri::ubyte_t> mem, yuri::size_t size, bool /*large*/)
//{
//	pBasicFrame f(new BasicFrame());
//	f->set_planes_count(1);
//	(*f)[0].set(mem,size);
//	return f;
//}
pBasicFrame BasicIOThread::duplicate_frame(pBasicFrame frame)
{
	pBasicFrame f = frame->get_copy();
	return f;
}

BasicIOThread::BasicIOThread(log::Log &log_,pwThreadBase parent, yuri::sint_t inp, yuri::sint_t outp, std::string id):
	ThreadBase(log_,parent),in_ports(inp),out_ports(outp),latency(200000),
	active_pipes(0),cpu_affinity(-1),fps_stats(0),pts_base(boost::posix_time::not_a_date_time)
{
	params.merge(*configure());
	log.set_label(std::string("[")+id+"] ");
	resize(inp,outp);
}

BasicIOThread::~BasicIOThread()
{
	close_pipes();
}

void BasicIOThread::run()
{
	IO_THREAD_PRE_RUN
	int ret;
	try {
		while (still_running()) {
			if (!active_pipes && (in_ports || !out_ports)) {
				//usleep(latency);
				ThreadBase::sleep(latency);
			}
			boost::this_thread::interruption_point();
			if (in_ports && !pipes_data_available()) {
#ifdef __linux__
				//log[verbose_debug] << "Requesting notifications" << "\n";
				request_notifications();
				//log[verbose_debug] << "Polling" << "\n";
				if (!(ret=poll(pipe_fds.get(),active_pipes,latency/1000))) continue;
				if (ret < 0) {
					switch (errno) {
						case EBADF: log[warning] << "Wrong fd set in fdset, file: "
						<< __FILE__ << ", line " << __LINE__ <<"\n";
						break;
						case EINTR: continue;
						default: log[warning] << "Error " << errno <<" ("<<
						strerror(errno)<< ") while reading from socket in file: "
						 << __FILE__ << ", line " << __LINE__ <<"\n";
						break;
					}
					continue;
				}
				read_notification();
#else
				ThreadBase::sleep(latency>>2);
#endif
			}
			log[verbose_debug] << "Stepping" << "\n";
			if (!step()) break;
		}
	}
	catch (boost::thread_interrupted &e)
	{
		log[debug] << "Thread interrupted" << "\n";
	}
	IO_THREAD_POST_RUN
}

void BasicIOThread::connect_in(yuri::sint_t index,shared_ptr<BasicPipe> pipe)
{
	if (index < 0 || static_cast<yuri::uint_t>(index) >= in.size()) throw exception::OutOfRange("Input pipe out of Range");
#ifdef BASICIOTHREAD_ENABLE_PORT_LOCK
	DEBUG_LOCKS
	boost::mutex::scoped_lock l(port_lock);
#endif
	if (in[index]) {
		log[debug] << "Disconnecting already connected pipe from in port "
				<< index << "\n";
		//in[index]->cancel_notifications(); // Just in case
	}
	in[index]=PipeConnector(pipe,get_this_ptr());
#ifdef BASICIOTHREAD_ENABLE_PORT_LOCK
	l.unlock();
#endif
	set_fds();

}
void BasicIOThread::connect_out(yuri::sint_t index,shared_ptr<BasicPipe> pipe)
{
	if (index < 0 || static_cast<yuri::uint_t>(index) >= out.size()) throw exception::OutOfRange("Output pipe out of Range");
#ifdef BASICIOTHREAD_ENABLE_PORT_LOCK
	DEBUG_LOCKS
	boost::mutex::scoped_lock l(port_lock);
#endif
	if (out[index]) log[debug] << "Disconnecting already connected pipe from out port " << index << "\n";
	out[index]=PipeConnector(pipe,get_this_ptr());
}

void BasicIOThread::resize(yuri::sint_t inp, yuri::sint_t outp)
{

#ifdef BASICIOTHREAD_ENABLE_PORT_LOCK
	DEBUG_LOCKS
	boost::mutex::scoped_lock l(port_lock);
#endif
	log[debug] << "Resizing to " << inp << " input ports and " << outp << " output ports." << "\n";
	in_ports=inp;
	out_ports=outp;
	//shared_ptr<BasicPipe> null_pipe;
	weak_ptr<ThreadBase> emptyptr;
	PipeConnector pc(emptyptr);
	in.resize(inp,pc);
	out.resize(outp,pc);
	streamed_frames.resize(outp,0);
	first_frame.resize(outp);
	//if (pipe_fds) delete [] pipe_fds;
#ifdef __linux__
	pipe_fds.reset(new struct pollfd[inp]);
	for (yuri::sint_t i=0;i<inp;++i) pipe_fds[i].events=0;
#endif
	log[debug] << "Konec resize" << "\n";
}

void BasicIOThread::close_pipes()
{

#ifdef BASICIOTHREAD_ENABLE_PORT_LOCK
	DEBUG_LOCKS
	boost::mutex::scoped_lock l(port_lock);
#endif
	log[debug] << "Closing pipes!" << "\n";
	for (yuri::sint_t i=0;i<out_ports;++i) {
		shared_ptr<BasicPipe> p=out[i];
		if (!p.get()) continue;
		p->close();
		out[i].reset();
	}
	/*for (int i=0;i<in_ports;++i) {
		shared_ptr<BasicPipe> p=in[i];
		if (!p.get()) continue;
		if (p->is_closed()) {
			//delete p;
			in[i].reset();
		}
	}*/
}
bool BasicIOThread::step()
{
	return false;
}


int BasicIOThread::set_fds()
{
#ifdef BASICIOTHREAD_ENABLE_PORT_LOCK
	DEBUG_LOCKS
	boost::mutex::scoped_lock l(port_lock);
#endif
	active_pipes=0;
	for (int i=0;i<in_ports;++i) {
		if (in[i]) {
#ifdef __linux__
			//in[i]->get_notification_fd();
			pipe_fds[active_pipes].fd=in[i]->get_notification_fd();
//			log[verbose_debug] << "Got fd " << pipe_fds[active_pipes].fd
//				<< " from pipe " << active_pipes<< "\n";
			pipe_fds[active_pipes].events=POLLIN|POLLPRI;
#endif
			++active_pipes;
		}
	}
	return active_pipes;
}

void BasicIOThread::request_notifications()
{
	/*for (int i=0;i<in_ports;++i) {
			if (in[i]) {
				log[verbose_debug] << "Requesting notifications from pipe " << i << "\n";
				in[i]->request_notify();
			}
	}*/
	set_fds();
}

bool BasicIOThread::pipes_data_available()
{
#ifdef BASICIOTHREAD_ENABLE_PORT_LOCK
	DEBUG_LOCKS
	boost::mutex::scoped_lock l(port_lock);
#endif
	for (yuri::sint_t i=0;i<in_ports;++i)
		if (in[i]) {
			if (in[i]->is_closed()) in[i].reset();
			else if (!in[i]->is_empty()) {
				//log[verbose_debug] << "Data available in pipe " << i << "\n";
				return true;
			}
		}
	return false;
}


void BasicIOThread::read_notification()
{
	char c;
	int dummy YURI_UNUSED;
	for (yuri::uint_t i=0;i<active_pipes;++i) {
#ifdef __linux__
		if (pipe_fds[i].revents&(POLLIN|POLLPRI)) {
			//log[verbose_debug] << "Reading notification from pipe " << i << "\n";
			dummy = read(pipe_fds[i].fd,&c,1);
			pipe_fds[i].revents=0;

			break;
		}
#endif
	}
}

bool BasicIOThread::push_raw_frame(yuri::sint_t index, pBasicFrame frame)
{
	assert(frame.get());
	if (index >= out_ports) return false;
	if (!out[index]) return false;
	out[index]->push_frame(frame);
	if (fps_stats) {
		if (streamed_frames[index]>=fps_stats) {
			boost::posix_time::ptime end_time = boost::posix_time::microsec_clock::local_time();
			boost::posix_time::time_duration delta= end_time - first_frame[index];
			double fps = 1.0e6 * static_cast<double>(streamed_frames[index]) / static_cast<double>(delta.total_microseconds());
			log[info] << "(output " << index << ") Streamed " << streamed_frames[index] << " in " << boost::posix_time::to_simple_string(delta) << ". That's " << fps << "fps" <<"\n";
			streamed_frames[index]=0;
		}
		if (!streamed_frames[index]) first_frame[index] = boost::posix_time::microsec_clock::local_time();
		streamed_frames[index]++;
	}
	return true;
}
bool BasicIOThread::push_raw_video_frame(yuri::sint_t index, pBasicFrame frame)
{
	assert(frame.get());
	if (index >= out_ports) return false;
	if (!out[index]) return false;
	out[index]->set_type(YURI_TYPE_VIDEO);
	return push_raw_frame(index,frame);
}

bool BasicIOThread::push_raw_audio_frame(yuri::sint_t index, pBasicFrame frame)
{
	assert(frame.get());
	if (index >= out_ports) return false;
	if (!out[index]) return false;
	out[index]->set_type(YURI_TYPE_AUDIO);
	return push_raw_frame(index,frame);
}


bool BasicIOThread::push_video_frame (yuri::sint_t index, pBasicFrame frame, yuri::format_t format, yuri::size_t width, yuri::size_t height, yuri::size_t pts, yuri::size_t duration, yuri::size_t dts)
{
	assert(frame.get());
//	log[verbose_debug] << "Setting format " << BasicPipe::get_format_string(format) << endl;
	frame->set_parameters(format,width,height);
	frame->set_time(pts,dts,duration);
	return push_raw_video_frame(index,frame);
}

bool BasicIOThread::push_video_frame (yuri::sint_t index, pBasicFrame frame, yuri::format_t format, yuri::size_t width, yuri::size_t height)
{
	assert(frame.get());
	// TODO: Set up values for PTS DTS and duration
	frame->set_parameters(format,width,height);
	return push_raw_video_frame(index,timestamp_frame(frame));
}

bool BasicIOThread::push_audio_frame (yuri::sint_t index, pBasicFrame frame, yuri::format_t format, yuri::usize_t channels, yuri::usize_t samples, yuri::size_t pts, yuri::size_t duration, yuri::size_t dts)
{
	assert(frame.get());
//	log[verbose_debug] << "Setting format " << BasicPipe::get_format_string(format) << "\n";
	frame->set_parameters(format,0,0,channels,samples);
	frame->set_time(pts,dts,duration);
	return push_raw_audio_frame(index,frame);
}
pBasicFrame BasicIOThread::timestamp_frame(pBasicFrame frame)
{
	if (pts_base==boost::posix_time::not_a_date_time) pts_base=boost::posix_time::microsec_clock::local_time();
	boost::posix_time::time_duration pts = boost::posix_time::microsec_clock::local_time() - pts_base;
	frame->set_time(pts.total_microseconds(),frame->get_dts(),frame->get_duration());
	return frame;
}
// TODO Stub, not implemented!!!!!!
pBasicFrame BasicIOThread::get_frame_as(yuri::sint_t index, yuri::format_t format)
{
	pBasicFrame frame;
	if (index>=in_ports || !in[index] || in[index]->is_empty()) return frame;
	if (format == YURI_FMT_NONE) return in[index]->pop_frame();

	return pBasicFrame();
}
void BasicIOThread::set_affinity(yuri::ssize_t affinity)
{
	cpu_affinity = affinity;
}
bool BasicIOThread::set_params(Parameters &parameters)
{
	std::pair<std::string,shared_ptr<Parameter> > par;
	BOOST_FOREACH(par,(parameters.params)) {
		if (par.second && !set_param(*(par.second))) return false;
	}
	return true;
}
/* Sets one parameter.
 *
 * @param:  parameter - parameter to set
 * @return false on error (like value can't be set) and true otherwise (even for unknown parameter)
 *
 */
bool BasicIOThread::set_param(const Parameter &parameter)
{
	if (parameter.name == "cpu") {
		cpu_affinity=parameter.get<yuri::ssize_t>();
	} else if (parameter.name == "fps_stats") {
		fps_stats = parameter.get<yuri::size_t>();
	} else if (parameter.name == "debug") {
		int debug;
		debug = parameter.get<yuri::sint_t>();
		if (debug) {
			yuri::sint_t orig = log.get_flags();
			log.set_flags(((orig&flag_mask)<<debug)|(orig&~flag_mask));
		}
	}
	return true;
}

pBasicFrame BasicIOThread::allocate_empty_frame(yuri::format_t format, yuri::size_t width, yuri::size_t height, bool large)
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
		//shared_array<yuri::ubyte_t> mem = allocate_memory_block(planesize,large);
//		(*pic)[i].set(mem,planesize);
		pic->get_plane(i).resize(planesize);
	}
	pic->set_parameters(format,width,height);
	return pic;
}

bool BasicIOThread::connect_threads(shared_ptr<BasicIOThread> src, yuri::sint_t s_idx, shared_ptr<BasicIOThread> target, yuri::sint_t t_idx, Log &log,std::string name, shared_ptr<Parameters> params)
{
	shared_ptr<Parameters> p = BasicPipe::configure();
	if (params) p->merge(*params);
	shared_ptr<BasicPipe> pipe = BasicPipe::generator(log,name,*p);
	target->connect_in(s_idx,pipe);
	src->connect_out(t_idx,pipe);
	return true;
}
}
}

// End of File
