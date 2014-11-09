/*!
 * @file 		yuri2.cpp
 * @author 		Zdenek Travnicek
 * @date 		4.8.2010
 * @date		21.11.2013
 * @date		11.9.2014
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2014
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef YURI_VERSION
#define YURI_VERSION "Unknown"
#endif
#define BOOST_LIB_DIAGNOSTIC


#include "yuri/yuri_listings.h"
#include "yuri/try_conversion.h"

#include "yuri/core/thread/XmlBuilder.h"
#include "yuri/exception/Exception.h"
#include "yuri/core/thread/FixedMemoryAllocator.h"

#include "yuri/version.h"
#include <iostream>
#include <memory>
#include <exception>

#include "yuri/core/frame/raw_frame_types.h"
#ifdef HAVE_BOOST_PROGRAM_OPTIONS
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#endif

// Defined as global variables, so these can be used in signal handler.
yuri::shared_ptr<yuri::core::XmlBuilder> builder;
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


void version()
{
	l[yuri::log::fatal] << "libyuri version " << yuri::yuri_version;

}
int main(int argc, char**argv)
{
	using namespace yuri;
	std::vector<std::string> params;
	int verbosity = 0;
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
		("list,l",po::value<std::string>()->implicit_value("classes"),"List registered classes (accepted values: classes, functions, formats, datagram_sockets, stream_sockets, pipes, converters)")
		("class,L",po::value<std::string>(),"List details of a single class")
		("convert,C",po::value<std::string>(), "Find conversion between format F1 and F2. Use syntax F1:F2.")
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
	if (vm.count("list") || vm.count("class")) {
		builder.reset(new core::XmlBuilder(l, core::pwThreadBase(), filename, arguments, true ));
		log::Log l_(std::cout);
		l_.set_flags(log::info);
		l_.set_quiet(true);
		if (vm.count("class")) {
			std::string class_name = vm["class"].as<std::string>();
			yuri::app::list_single_class(l_, class_name, verbosity);
		} else {
			std::string list_what = vm["list"].as<std::string>();
			yuri::app::list_registered_items(l_, list_what, verbosity);
		}
		return 0;
	}
	if (vm.count("convert")) {
		builder.reset(new core::XmlBuilder(l, core::pwThreadBase(), filename, arguments, true ));
		log::Log l_(std::cout);
		l_.set_flags(log::info);
		l_.set_quiet(true);
		std::string fmts = vm["convert"].as<std::string>();
		auto idx = fmts.find(':');
		if (idx == std::string::npos) {
			l[log::fatal] << "Formats specified wrongly";
		} else {
			yuri::app::try_conversion(l_, fmts.substr(0,idx), fmts.substr(idx+1));
		}
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

				yuri::app::list_registered_items(l_, list_what, verbosity);
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
		l[log::fatal] << "Usage: " << argv[0] << " " << filename << reqs;
		l[log::fatal] <<"  ";
		l[log::fatal] << "Variables:";
		for (const auto& var: vars) {
			std::string filler(20-var.name.size(),' ');
			l[log::fatal] << var.name << ":" << filler << var.description
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
	auto mp = yuri::core::FixedMemoryAllocator::clear_all();
	l[log::info] << "Memory pool cleared ("<< mp.first << " blocks, " << mp.second << " bytes)";
	return 0;
}

