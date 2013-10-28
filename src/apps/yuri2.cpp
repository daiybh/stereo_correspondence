/*!
 * @file 		yuri.cpp
 * @author 		Zdenek Travnicek
 * @date 		4.8.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef YURI_VERSION
#define YURI_VERSION "Unknown"
#endif
#define BOOST_LIB_DIAGNOSTIC
#include "yuri/core/thread/XmlBuilder.h"
#include "yuri/version.h"
#include <iostream>
#include <memory>
#include <exception>
#include "yuri/core/parameter/Parameters.h"
#include "yuri/core/pipe/Pipe.h"
#include "yuri/core/thread/IOThreadGenerator.h"
#include "yuri/core/socket/DatagramSocketGenerator.h"
#include "yuri/core/socket/StreamSocketGenerator.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/compressed_frame_params.h"
#include "yuri/core/frame/raw_audio_frame_params.h"
#include "yuri/event/BasicEventConversions.h"
#include "yuri/core/pipe/PipeGenerator.h"

#ifdef HAVE_BOOST_PROGRAM_OPTIONS
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#endif

yuri::shared_ptr<yuri::core::XmlBuilder> builder;
int verbosity = 0;
yuri::log::Log l(std::clog);


#if defined YURI_LINUX || defined YURI_APPLE
#include <signal.h>
#include <string.h>
static void sigHandler(int sig, siginfo_t *siginfo, void *context);
static struct sigaction act;


void sigHandler(int sig, siginfo_t */*siginfo*/, void */*context*/)
{
#if !defined YURI_APPLE
	if (sig==SIGRTMIN) {
		l[yuri::log::warning] << "Realtime signal 0! Ignoring...";
		return;
	}
#endif
	if (builder)
		builder->request_end(yuri::core::yuri_exit_interrupted);
	act.sa_handler = SIG_DFL;
	act.sa_flags &= ~SA_SIGINFO;
	sigaction(SIGINT,&act,0);
}
#endif


#ifdef HAVE_BOOST_PROGRAM_OPTIONS
void usage(const po::options_description& options)
{
	//l.set_quiet(true);
	l[yuri::log::fatal]
			<< "Usage:	yuri [options] [-i] <file> [[-p] params...]\n\n"
			<< options;
}
#endif
//
void list_params(yuri::log::Log& l_, const yuri::core::Parameters& params)
{
	using namespace yuri;
	for (const auto& p: params) {
		const auto& param = p.second;
		const std::string& pname = param.get_name();
		if (pname[0] != '_') {
			l_[log::info] << "\t\t"
				<< std::setfill(' ') << std::left << std::setw(20)
				<< (pname + ": ")
				<< std::right << std::setw(10) << param.get<std::string>();
			const std::string& d = param.get_description();
			if (!d.empty()) {
				l_[log::info] << "\t\t\t" << d;
			}
		}
	}
}

void list_registered(yuri::log::Log& l_)
{
	using namespace yuri;
	if (verbosity>=0) l_[log::info]<<"List of registered objects:" ;
	auto& generator = yuri::IOThreadGenerator::get_instance();
	for (const auto& name: generator.list_keys()) {
		if (verbosity < 0) {
			l_[log::info] << name;
		} else {
			l_[log::info] /*<< "Module: "*/ << "..:: " << name << " ::..";
			const auto& params = generator.configure(name);
			const std::string& desc = params.get_description();
			if (!desc.empty()) {
				l_[log::info] << "\t" << desc;
			}
			list_params(l_, params);
			l_[log::info];
		}
	}

}

namespace {

template<class T, class Stream>
void print_array(Stream& a, const std::vector<T>& data, const std::string& title)
{
	if (data.size()) {
		a << "\t"<< title << ":";
		for (const auto& sn: data) {
			a << " " << sn;
		}
	}
}

template<class T>
void print_array(yuri::log::Log& l_, const std::vector<T>& data, const std::string& title)
{
	auto a = l_[yuri::log::info];
	print_array(a, data, title);
}
}

void list_formats(yuri::log::Log& l_)
{
	using namespace yuri;
	l_[log::info] << "List of registered formats:";
	l_[log::info] << "+++ RAW FORMATS +++";
	for (const auto& fmt: core::raw_format::formats())
	{
		const auto& info = fmt.second;
		l_[log::info] << "\"" << info.name << "\"";
		print_array(l_, info.short_names, "Short names");
		auto a = l_[log::info];
		a << "\tPlanes " << info.planes.size();
		for (const auto& pi: info.planes) {
			float bps = static_cast<float>(pi.bit_depth.first)/pi.bit_depth.second;
			a << ", " << pi.components << ": " << bps << "bps";
		}
	}

	l_[log::info] << "\t";
	l_[log::info] << "+++ COMPRESSED FORMATS +++";
	for (const auto& fmt: core::compressed_frame::formats())
	{
		const auto& info = fmt.second;
		l_[log::info] << "\"" << info.name << "\"";
		print_array(l_, info.short_names, "Short names");
		print_array(l_, info.mime_types, "Mime types");
		if (!info.fourcc.empty()) {
			l_[log::info] << "FOURCC: " << info.fourcc;
		}
	}

	l_[log::info] << "\t";
	l_[log::info] << "+++ RAW AUDIO FORMATS +++";
	for (const auto& fmt: core::raw_audio_format::formats())
	{
		const auto& info = fmt.second;
		l_[log::info] << "\"" << info.name << "\"";
		print_array(l_, info.short_names, "Short names");

		l_[log::info] << "Bits per sample: " << info.bits_per_sample;
		l_[log::info] << "Endianness: " << (info.little_endian?"little":"big");
	}
}
void list_dgram_sockets(yuri::log::Log& l_)
{
	using namespace yuri;
	l_[log::info] << "List of registered datagram_socket implementations:";
	const auto& reg = core::DatagramSocketGenerator::get_instance();
	for (const auto& sock: reg.list_keys())
	{
		l_[log::info] << sock;

	}
}
void list_stream_sockets(yuri::log::Log& l_)
{
	using namespace yuri;
	l_[log::info] << "List of registered datagram_socket implementations:";
	const auto& reg = core::StreamSocketGenerator::get_instance();
	for (const auto& sock: reg.list_keys())
	{
		l_[log::info] << sock;

	}
}
namespace {
using namespace yuri;
std::map<event::event_type_t, std::string> event_names=
{
		{event::event_type_t::bang_event, 		"BANG"},
		{event::event_type_t::boolean_event, 	"boolean"},
		{event::event_type_t::integer_event, 	"integer"},
		{event::event_type_t::double_event, 	"double"},
		{event::event_type_t::string_event, 	"string"},
		{event::event_type_t::time_event, 		"time"},
		{event::event_type_t::vector_event, 	"vector"},
		{event::event_type_t::dictionary_event,	"dictionary"},
		{event::event_type_t::undetermined_event,"undetermined"},

};
const std::string unknown_event_name {"??"};
const std::string& event_type_name(event::event_type_t e)
{
	auto it = event_names.find(e);
	if (it == event_names.end()) return unknown_event_name;
	return it->second;
}
}

void list_functions(yuri::log::Log& l_)
{
	using namespace yuri;
	l_[log::info] << "List of registered event functions:";
	auto& reg = event::EventFunctionsFactory::get_instance();
	std::map<std::string, std::vector<std::string>> signatures;
	for (const auto& func: reg.get_map())
	{
		const event::event_function_record_t& rec = func.second;
		//l_[log::info] << func.first;
		std::stringstream sstr;

		sstr << func.first << "(";
		bool first = true;
		for (const auto& x: rec.param_types) {
			if (first) {first = false;}
			else sstr << ", ";
			sstr << event_type_name(x);
		}
		sstr << ") -> " << event_type_name(rec.return_type);
		signatures[func.first].emplace_back(sstr.str());
	}
	for (const auto& name: signatures) {
		l_[log::info] << name.first;
		for (const auto& sig: name.second) {
			l_[log::info] << "\t" << sig;
		}
	}
}

void list_pipes(yuri::log::Log& l_)
{
	using namespace yuri;
	l_[log::info] << "List of registered pipe classes:";
	const auto& generator = core::PipeGenerator::get_instance();
	for (const auto& name: generator.list_keys()) {
		if (verbosity < 0) {
			l_[log::info] << name;
		} else {
			l_[log::info] /*<< "Module: "*/ << "..:: " << name << " ::..";
			const auto& params = generator.configure(name);
			const std::string& desc = params.get_description();
			if (!desc.empty()) {
				l_[log::info] << "\t" << desc;
			}
			list_params(l_, params);
			l_[log::info];
		}
	}
}
//void list_converters(Log l_)
//{
//	for (const auto& conv: core::RegisteredClass::get_all_converters()) {
//		l_[info] << "Convertors from " << core::BasicPipe::get_format_string(conv.first.first)
//		 << " to " << core::BasicPipe::get_format_string(conv.first.second) << std::endl;
//		for(const auto& c: conv.second) {
//			if (!c) std::cout << "??" <<std::endl;
//			l_[info] << "\t" << c->id << std::endl;
//		}
//	}
//}
void version()
{
	l[yuri::log::fatal] << "libyuri version " << yuri::yuri_version;

}
int main(int argc, char**argv)
{
	using namespace yuri;
	std::vector<std::string> params;
	//yuri::uint_t verbosity;
	std::string filename;
	std::vector<std::string> arguments;
	l.set_label("[YURI2] ");
	l.set_flags(log::info|log::show_level|log::use_colors);
	bool show_info = false;
#ifdef HAVE_BOOST_PROGRAM_OPTIONS
	po::options_description options("General options");
	options.add_options()
		("help,h","print help")
		("version,V","Show version of yuri and libyuri")
		("verbose,v","Show verbose output")
		("quiet,q","Limit output")
		("verbosity",po::value<int> (&verbosity)->default_value(0),"Verbosity level <-3, 4>")
		("input-file,f",po::value<std::string>(&filename),"Input XML file")
		("parameter,p",po::value<std::vector<std::string> >(&arguments),"Parameters to pass to libyuri builder")
		("list,l",po::value<std::string>()->implicit_value("classes"),"List registered classes (accepted values classes, functions, formats, datagram_sockets, pipes)")
		("app-info,a","Show info about XML file");



	po::positional_options_description p;
	p.add("input-file", 1);
	p.add("parameter", -1);

	po::variables_map vm;
	try {
		po::store(po::command_line_parser(argc, argv).options(options).positional(p).run(), vm);
		po::notify(vm);
	}
	catch (po::error &e) {
		l[log::fatal] << "Wrong options specified (" << e.what() <<")";
		usage(options);
		return 1;
	}

	if (verbosity < -3) verbosity = -3;
	if (verbosity >  4) verbosity =  4;
	if (vm.count("quiet")) verbosity=-1;
	else if (vm.count("verbose")) verbosity=1;

	int log_params = l.get_flags()&~log::flag_mask;
	//cout << "Verbosity: " << verbosity << std::endl;
	if (verbosity >=0)	l.set_flags((log::info<<(verbosity))|log_params);
	else l.set_flags((log::info>>(-verbosity))|log_params);
	//cout << "Verbosity: " << verbosity << ", flags: " << (l.get_flags()&flag_mask)<<std::endl;
	if (vm.count("help")) {
		l.set_quiet(true);
		usage(options);
		return -1;
	}
	if (vm.count("version")) {
		l.set_quiet(true);
		version();
		return 1;
	}
	if (vm.count("list")) {
		builder.reset(new core::XmlBuilder(l, core::pwThreadBase(), filename, arguments, true ));
		log::Log l_(std::cout);
		l_.set_flags(log::info);
		l_.set_quiet(true);
		std::string list_what = vm["list"].as<std::string>();
		if (iequals(list_what,"formats")) list_formats(l_);
		else if (iequals(list_what,"datagram_sockets") || iequals(list_what,"datagram")) list_dgram_sockets(l_);
		else if (iequals(list_what,"stream_sockets") || iequals(list_what,"stream")) list_stream_sockets(l_);
		else if (iequals(list_what,"functions")) list_functions(l_);
		else if (iequals(list_what,"pipes")) list_pipes(l_);
		else list_registered(l_);
		return 0;
	}
	if (vm.count("app-info")) {
		show_info=true;
		l.set_flags(log::fatal);
		l.set_quiet(true);
	}
#else
	for (int i=1;i<argc;++i) {
		if (argv[i][0]=='-') {
			if (iequals(std::string(argv[i]+1),"l")) {
				std::string list_what("classes");
				if (i<argc-1) {
					list_what=argv[++i];
				}
				builder.reset(new core::XmlBuilder(l, core::pwThreadBase(), filename, arguments, true ));
				log::Log l_(std::cout);
				l_.set_flags(log::info);
				l_.set_quiet(true);
				if (iequals(list_what,"formats")) list_formats(l_);
				else if (iequals(list_what,"datagram_sockets") || iequals(list_what,"datagram")) list_dgram_sockets(l_);
				else if (iequals(list_what,"stream_sockets") || iequals(list_what,"stream")) list_stream_sockets(l_);
				else if (iequals(list_what,"functions")) list_functions(l_);
				else if (iequals(list_what,"pipes")) list_pipes(l_);
				else list_registered(l_);
			}
		} else {
			if (filename.empty()) filename = argv[i];
			else arguments.push_back(argv[i]);
		}
	}

#endif

	if (filename.empty()) {
		l[log::fatal] << "No input file specified";
#ifdef HAVE_BOOST_PROGRAM_OPTIONS
		usage(options);
#endif
		return -1;
	}

	l[log::debug] << "Loading file " << filename;
	try {
		builder.reset( new core::XmlBuilder (l, core::pwThreadBase(),filename, arguments, show_info));
	}
	catch (exception::Exception &e) {
		l[log::fatal] << "failed to initialize application: " << e.what();
		return 1;
	}
	catch (std::exception &e) {
		l[log::fatal] << "An error occurred during initialization: " << e.what();
		return 1;
	}

	if (show_info) {
		const auto& vars = builder->get_variables();
		l[log::fatal] << "Application " << builder->get_app_name();
		l[log::fatal] << "  ";
		const std::string desc = builder->get_description();
		if (!desc.empty()) l[log::fatal] << "Description: " << desc;
		std::string reqs;
//		for (const auto& it: vars) {
//			if (it->second->required) reqs=reqs + " " + it->second->name + "=<value>";
//		}
		l[log::fatal] << "Usage: " << argv[0] << " " << filename << reqs;
		l[log::fatal] <<"  ";
		l[log::fatal] << "Variables:";
		for (const auto& var: vars) {
//			const auto& var = it->second;
			std::string filler(20-var.name.size(),' ');
			l[log::fatal] << var.name << ":" << filler << var.description
//					<< " [default value: " << var->def
					<< " [value: " << var.value << "]";
		}
		l[log::fatal] <<"  ";
		return 0;
	}

#if defined YURI_LINUX || defined YURI_APPLE
	memset (&act, '\0', sizeof(act));
	act.sa_sigaction = &sigHandler;
	act.sa_flags = SA_SIGINFO;
	sigaction(SIGINT,&act,0);
#if !defined YURI_APPLE
	sigaction(SIGRTMIN,&act,0);
#endif
#endif
	try {
		(*builder)();
		l[log::info] << "Application successfully finished";
	}
	catch (yuri::exception::Exception &e) {
		l[log::fatal] << "Application failed to start: " << e.what();
	}
	catch(std::exception &e) {
		l[log::fatal] << "An error occurred during execution: " << e.what();
	}
	// Explicit release of resources is needed here, so the destruction can take place before main ends
	// Otherwise it would be destroyed among global variables and this could lead to segfaults.
	builder.reset();
	l[log::info] << "Application successfully destroyed";
	return 0;
}
