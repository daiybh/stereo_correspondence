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
#include "yuri/core/ApplicationBuilder.h"
#include "yuri/version.h"
#include <iostream>
#include <memory>
#include <exception>
#include <signal.h>
#include <boost/foreach.hpp>
#include <boost/algorithm/string.hpp>
#include "yuri/core/Parameters.h"
#include "yuri/core/RegisteredClass.h"
#include "yuri/core/BasicPipe.h"
//using namespace std;
using namespace yuri;
using namespace yuri::log;
using boost::iequals;
using yuri::shared_ptr;
#include <boost/program_options.hpp>
namespace po = boost::program_options;

static yuri::shared_ptr<core::ApplicationBuilder> b;
static po::options_description options("General options");
static int verbosity;
static Log l(std::clog);


#ifdef __linux__
static void sigHandler(int sig, siginfo_t *siginfo, void *context);
static struct sigaction act;


void sigHandler(int /*sig*/, siginfo_t */*siginfo*/, void */*context*/)
{
	if (b)
		b->request_end();
	act.sa_handler = SIG_DFL;
	act.sa_flags &= ~SA_SIGINFO;

	sigaction(SIGINT,&act,0);
}
#endif



void usage()
{
	l.set_quiet(true);
	l[fatal]
			<< "Usage:	yuri [options] [-i] <file> [[-p] params...]" << std::endl << std::endl
			<< options << std::endl;
}

void list_registered(Log l_)
{
	l_[info]<<"List of registered objects:" << std::endl;
std::string name;
	shared_ptr<std::vector<std::string> > v = yuri::core::RegisteredClass::list_registered();
	BOOST_FOREACH(name,*v) {
		if (verbosity>=0)
			l_[fatal] << "..:: " << name << " ::.." << std::endl;
		else l_[fatal] << name << std::endl;
		core::pParameters p = core::RegisteredClass::get_params(name);
		if (!p) l_[info] << "\t\tclass has no configuration defined!" << std::endl;
		else {
			if (!p->get_description().empty()) l_[info]<< "\t"
					<< p->get_description() << std::endl;
			long fmt;
			if (p->get_input_formats().size()) {
				BOOST_FOREACH(fmt,p->get_input_formats()) {
					l_[normal]<< "\t\tSupports input format: " << core::BasicPipe::get_format_string(fmt) << std::endl;
				}
			} else l_[normal] << "\t\tClass does not have any restrictions on input pipes defined." << std::endl;
			if (p->get_output_formats().size()) {
				BOOST_FOREACH(fmt,p->get_output_formats()) {
					l_[normal]<< "\t\tSupports output format: " << core::BasicPipe::get_format_string(fmt) << std::endl;
				}
			} else l_[normal] << "\t\tClass does not have any restrictions on output pipes defined." << std::endl;
			if (p->params.size()) {
				std::pair<std::string,core::pParameter > par;
				BOOST_FOREACH(par,p->params) {
					l_[info] << "\t\t'" << par.first << "' has default value \""
							<< par.second->get<std::string>() << "\"" << std::endl;
					if (!par.second->description.empty()) l_[info] << "\t\t\t"
							<< par.second->description << std::endl;
				}
			} else l_[info] << "\t\tClass has no parameters" << std::endl;
		}
	}
}
void list_formats(Log& l_)
{
	l_[info] << "List of registered formats:" << std::endl;
std::string name;
	boost::mutex::scoped_lock lock(core::BasicPipe::format_lock);
	std::pair<yuri::format_t, yuri::FormatInfo_t > fmtp;
	BOOST_FOREACH(fmtp, core::BasicPipe::formats) {
		yuri::FormatInfo_t fmt = fmtp.second;
		l_[fatal] << fmt->long_name << std::endl;
		if (fmt->short_names.size()) {
			bool f = true;
		std::stringstream ss;
			ss << "\tAvailable as: ";
			BOOST_FOREACH(std::string s,fmt->short_names) {
				if (!f) ss << ", ";
				f=false;
				ss << s;
			}
			l_[info] << "" << (ss.str()) << std::endl;
		}
		if (fmt->mime_types.size()) {
			bool f = true;
		std::stringstream ss;
			ss << "\tUses mime types: ";
			BOOST_FOREACH(std::string s,fmt->mime_types) {
				if (!f) ss << ", ";
				f=false;
				ss << s;
			}
			l_[info] << "" << (ss.str()) << std::endl;
		}
	}

}

void list_converters(Log l_)
{
	std::pair<std::pair<long,long>,std::vector<shared_ptr<core::Converter> > > conv;
	BOOST_FOREACH(conv,core::RegisteredClass::get_all_converters()) {
		l_[info] << "Convertors from " << core::BasicPipe::get_format_string(conv.first.first)
		 << " to " << core::BasicPipe::get_format_string(conv.first.second) << std::endl;
		shared_ptr<core::Converter> c;
		BOOST_FOREACH(c,conv.second) {
			if (!c) std::cout << "??" <<std::endl;
			l_[info] << "\t" << c->id << std::endl;
		}
	}
}
void version()
{
	//l[fatal] << "yuri version " << YURI_VERSION << std::endl;
	l[fatal] << "libyuri version " << yuri::yuri_version << std::endl;

}
int main(int argc, char**argv)
{
	std::vector<std::string> params;
	//yuri::uint_t verbosity;
	std::string filename;
	std::vector<std::string> arguments;
	l.setLabel("[YURI] ");
	l.setFlags(info);
	options.add_options()
		("help,h","print help")
		("version,V","Show version of yuri and libyuri")
		("verbose,v","Show verbose output")
		("quiet,q","Limit output")
		("verbosity",po::value<int> (&verbosity)->default_value(0),"Verbosity level <-3, 4>")
		("input-file,f",po::value<std::string>(&filename),"Input XML file")
		("parameter,p",po::value<std::vector<std::string> >(&arguments),"Parameters to pass to libyuri builder")
		("list,l",po::value<std::string>()->implicit_value("classes"),"List registered objects/formats");



	po::positional_options_description p;
	p.add("input-file", 1);
	p.add("parameter", -1);

	po::variables_map vm;
	try {
		po::store(po::command_line_parser(argc, argv).options(options).positional(p).run(), vm);
		po::notify(vm);
	}
	catch (po::error &e) {
		l[fatal] << "Wrong options specified (" << e.what() <<")"<< std::endl;
		usage();
		return 1;
	}

	if (verbosity < -3) verbosity = -3;
	if (verbosity >  4) verbosity =  4;
	if (vm.count("quiet")) verbosity=-1;
	else if (vm.count("verbose")) verbosity=1;

	//cout << "Verbosity: " << verbosity << std::endl;
	if (verbosity >=0)	l.setFlags((info<<(verbosity)));
	else l.setFlags((info>>(-verbosity)));
	//cout << "Verbosity: " << verbosity << ", flags: " << (l.get_flags()&flag_mask)<<std::endl;
	if (vm.count("help")) {
		l.set_quiet(true);
		usage();
		return -1;
	}
	if (vm.count("version")) {
		l.set_quiet(true);
		version();
		return 1;
	}
	if (vm.count("list")) {
		b.reset( new core::ApplicationBuilder (l,core::pwThreadBase()));
		b->find_modules();
		b->load_modules();
		l.set_quiet(true);
	std::string list_what = vm["list"].as<std::string>();
		Log l_(std::cout);
		l_.setFlags(l.get_flags());
		l_.set_quiet(true);
		l_[debug] << "Listing " << list_what <<std::endl;
		if (iequals(list_what,"classes")) list_registered(l_);
		else if (iequals(list_what,"formats")) list_formats(l_);
		else if (iequals(list_what,"converters")) list_converters(l_);
		else std::cout << "Wrong value for --list parameter" << std::endl;

		exit(0);
	}
	/*if (iequals(argv[1],"--converters")) {

		exit(0);
	}*/

	if (filename.empty()) {
		l[fatal] << "No input file specified" << std::endl;
		usage();
		return -1;
	}

	l[debug] << "Loading file " << filename << std::endl;
	try {
		b.reset( new core::ApplicationBuilder (l, core::pwThreadBase(),filename,arguments));
	}
	catch (exception::Exception &e) {
		l[fatal] << "failed to initialize application: " << e.what() << std::endl;
		exit(1);
	}
	catch (std::exception &e) {
		l[fatal] << "An error occurred during initialization: " << e.what() << std::endl;
		exit(1);
	}
#ifdef __linux__
	memset (&act, '\0', sizeof(act));
	act.sa_sigaction = &sigHandler;
	act.sa_flags = SA_SIGINFO;
	sigaction(SIGINT,&act,0);
#endif
	try {
		(*b)();
	}
	catch (yuri::exception::Exception &e) {
		l[fatal] << "Application failed to start: " << e.what() << std::endl;
	}
	catch(std::exception &e) {
		l[fatal] << "An error occurred during execution: " << e.what() << std::endl;
	}
	return 0;
}
