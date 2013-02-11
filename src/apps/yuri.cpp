/*
 * yuri.cpp
 *
 *  Created on: Aug 4, 2010
 *      Author: neneko
 */
#ifndef YURI_VERSION
#define YURI_VERSION "Unknown"
#endif

#include "yuri/config/ApplicationBuilder.h"
//#include "yuri/version.h"
#include <iostream>
#include <memory>
#include <exception>
#include <signal.h>
#include <boost/foreach.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace yuri::io;
using namespace yuri::log;
using namespace yuri::config;
using boost::iequals;
#include <boost/program_options.hpp>
namespace po = boost::program_options;

static shared_ptr<ApplicationBuilder> b;
static po::options_description options("General options");
static int verbosity;
static Log l(cout);


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
	l[fatal]
			<< "Usage:	yuri [options] [-i] <file> [[-p] params...]" << endl << endl
			<< options << endl;
}

void list_registered()
{
	l[info]<<"List of registered objects:" << endl;
	string name;
	shared_ptr<vector<string> > v = yuri::config::RegisteredClass::list_registered();
	BOOST_FOREACH(name,*v) {
		if (verbosity>=0)
			l[fatal] << "..:: " << name << " ::.." << endl;
		else l[fatal] << name << endl;
		shared_ptr<Parameters> p = RegisteredClass::get_params(name);
		if (!p) l[info] << "\t\tclass has no configuration defined!" << endl;
		else {
			if (!p->get_description().empty()) l[info]<< "\t"
					<< p->get_description() << endl;
			long fmt;
			if (p->get_input_formats().size()) {
				BOOST_FOREACH(fmt,p->get_input_formats()) {
					l[normal]<< "\t\tSupports input format: " << BasicPipe::get_format_string(fmt) << endl;
				}
			} else l[normal] << "\t\tClass does not have any restrictions on input pipes defined." << endl;
			if (p->get_output_formats().size()) {
				BOOST_FOREACH(fmt,p->get_output_formats()) {
					l[normal]<< "\t\tSupports output format: " << BasicPipe::get_format_string(fmt) << endl;
				}
			} else l[normal] << "\t\tClass does not have any restrictions on output pipes defined." << endl;
			if (p->params.size()) {
				pair<string,shared_ptr<Parameter> > par;
				BOOST_FOREACH(par,p->params) {
					l[info] << "\t\t'" << par.first << "' has default value \""
							<< par.second->get<string>() << "\"" << endl;
					if (!par.second->description.empty()) l[info] << "\t\t\t"
							<< par.second->description << endl;
				}
			} else l[info] << "\t\tClass has no parameters" << endl;
		}
	}
}
void list_formats()
{
	l[info] << "List of registered formats:" << endl;
	string name;
	boost::mutex::scoped_lock lock(BasicPipe::format_lock);
	std::pair<yuri::format_t, yuri::FormatInfo_t > fmtp;
	BOOST_FOREACH(fmtp, BasicPipe::formats) {
		yuri::FormatInfo_t fmt = fmtp.second;
		l[fatal] << fmt->long_name << endl;
		if (fmt->short_names.size()) {
			bool f = true;
			stringstream ss;
			ss << "\tAvailable as: ";
			BOOST_FOREACH(string s,fmt->short_names) {
				if (!f) ss << ", ";
				f=false;
				ss << s;
			}
			l[info] << "" << (ss.str()) << endl;
		}
		if (fmt->mime_types.size()) {
			bool f = true;
			stringstream ss;
			ss << "\tUses mime types: ";
			BOOST_FOREACH(string s,fmt->mime_types) {
				if (!f) ss << ", ";
				f=false;
				ss << s;
			}
			l[info] << "" << (ss.str()) << endl;
		}
	}

}

void list_converters()
{
	pair<pair<long,long>,vector<shared_ptr<Converter> > > conv;
	BOOST_FOREACH(conv,RegisteredClass::get_all_converters()) {
		cout << "Convertors from " << BasicPipe::get_format_string(conv.first.first)
		 << " to " << BasicPipe::get_format_string(conv.first.second) << endl;
		shared_ptr<Converter> c;
		BOOST_FOREACH(c,conv.second) {
			if (!c) cout << "??" <<endl;
			cout << "\t" << c->id << endl;
		}
	}
}
void version()
{
	l[fatal] << "yuri version " << YURI_VERSION << endl;
	//l[fatal] << "libyuri version " << yuri::get_yuri_version() << endl;

}
int main(int argc, char**argv)
{
	vector<string> params;
	//yuri::uint_t verbosity;
	string filename;
	vector<string> arguments;
	l.setLabel("[YURI] ");
	l.setFlags(info);
	options.add_options()
		("help,h","print help")
		("version,V","Show version of yuri and libyuri")
		("verbose,v","Show verbose output")
		("quiet,q","Limit output")
		("verbosity",po::value<int> (&verbosity)->default_value(0),"Verbosity level <-3, 4>")
		("input-file,f",po::value<string>(&filename),"Input XML file")
		("parameter,p",po::value<vector<string> >(&arguments),"Parameters to pass to libyuri builder")
		("list,l",po::value<string>()->implicit_value("classes"),"List registered objects/formats");



	po::positional_options_description p;
	p.add("input-file", 1);
	p.add("parameter", -1);

	po::variables_map vm;
	try {
		po::store(po::command_line_parser(argc, argv).options(options).positional(p).run(), vm);
		po::notify(vm);
	}
	catch (po::error &e) {
		l[fatal] << "Wrong options specified (" << e.what() <<")"<< endl;
		usage();
		return 1;
	}

	if (verbosity < -3) verbosity = -3;
	if (verbosity >  4) verbosity =  4;
	if (vm.count("quiet")) verbosity=-1;
	else if (vm.count("verbose")) verbosity=1;

	//cout << "Verbosity: " << verbosity << endl;
	if (verbosity >=0)	l.setFlags((info<<(verbosity)));
	else l.setFlags((info>>(-verbosity)));
	//cout << "Verbosity: " << verbosity << ", flags: " << (l.get_flags()&flag_mask)<<endl;
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
		b.reset( new ApplicationBuilder (l,pThreadBase()));
		b->find_modules();
		b->load_modules();
		l.set_quiet(true);
		string list_what = vm["list"].as<string>();
		l[debug] << "Listing " << list_what <<endl;
		if (iequals(list_what,"classes")) list_registered();
		else if (iequals(list_what,"formats")) list_formats();
		else if (iequals(list_what,"converters")) list_converters();
		else cout << "Wrong value for --list parameter" << endl;

		exit(0);
	}
	/*if (iequals(argv[1],"--converters")) {

		exit(0);
	}*/

	if (filename.empty()) {
		l << "No input file specified" << endl;
		usage();
		return -1;
	}

	l[debug] << "Loading file " << filename << endl;
	try {
		b.reset( new ApplicationBuilder (l,pThreadBase(),filename,arguments));
	}
	catch (Exception &e) {
		l[fatal] << "failed to initialize application: " << e.what() << endl;
		exit(1);
	}
	catch (std::exception &e) {
		l[fatal] << "An error occurred during initialization: " << e.what() << endl;
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
	catch (Exception &e) {
		l[fatal] << "Application failed to start: " << e.what() << endl;
	}
	catch(std::exception &e) {
		l[fatal] << "An error occurred during execution: " << e.what() << endl;
	}
	return 0;
}
