/*!
 * @file 		BasicIOMacros.h
 * @author 		Zdenek Travnicek
 * @date 		10.10.2011
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2011 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef BASICIOMACROS_H_
#define BASICIOMACROS_H_


#define IO_THREAD_INIT(name) params.merge(*configure()); \
	params.merge(parameters); \
	set_params(params);


#define IO_THREAD_CONSTRUCTOR /*throw (InitializationFailed)*/

#define IO_THREAD_GENERATOR_DECLARATION 	static shared_ptr<yuri::io::BasicIOThread> generate(yuri::log::Log &_log, yuri::io::pThreadBase parent,yuri::config::Parameters& parameters) /*throw (Exception)*/;
//#define IO_THREAD_GENERATOR(cls) shared_ptr<yuri::io::BasicIOThread> cls::generate(yuri:log::Log &_log,yuri::io::pThreadBase parent, yuri::config::Parameters& parameters) /*throw (Exception)*/
#define IO_THREAD_GENERATOR(cls) shared_ptr<yuri::io::BasicIOThread> cls::generate(yuri::log::Log &_log,yuri::io::pThreadBase parent, yuri::config::Parameters& parameters) /*throw (Exception)*/ \
{ \
	shared_ptr<cls> obj; \
	try { \
		obj.reset(new cls(_log,parent,parameters)); \
	} \
	catch (std::exception &e) { \
		throw yuri::exception::InitializationFailed(std::string(#cls) + "constuctor failed: " + e.what()); \
	} \
	return obj; \
}

#define IO_THREAD_PRE_RUN  \
	print_id(yuri::log::info); \
	if (cpu_affinity >= 0) bind_to_cpu(cpu_affinity);
#define IO_THREAD_POST_RUN \
	close_pipes();

//#define PLANE_SIZE(frame, plane) (*frame)[plane].get_size()
//#define PLANE_DATA(frame, plane) (*frame)[plane].data
//#define PLANE_RAW_DATA(frame, plane) (*frame)[plane].data.get()
#define PLANE_SIZE(frame, plane) (*frame)[plane].size()
#define PLANE_DATA(frame, plane) (*frame)[plane]
#define PLANE_RAW_DATA(frame, plane) &((*frame)[plane][0])

#endif /* BASICIOMACROS_H_ */
