/*!
 * @file 		test.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		9.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include <iostream>
#include "yuri/log/Log.h"
#include "pipe/PipeGenerator.h"
#include "frame/RawVideoFrame.h"
#include "yuri/core/thread/IOThread.h"
#include "yuri/core/parameter/Parameter.h"
#include "yuri/core/thread/IOThreadGenerator.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/thread/FixedMemoryAllocator.h"
#include "yuri/core/utils/ModuleLoader.h"
#include "yuri/core/thread/XmlBuilder.h"

yuri::log::Log l(std::clog);
using namespace yuri;

class IO: public core::IOThread
{
public:

	IOTHREAD_GENERATOR_DECLARATION
	IO(const log::Log& log_, core::pwThreadBase parent, const core::Parameters& params):
		IOThread(log_, parent, 1, 1, "IO") {
		log[log::info] << "IO::IO()";
		for (const auto& p: params) {
			log[log::info] << "Got param " << p.first << " = " << p.second.get<std::string>();
		}
	}
	~IO() noexcept{log[log::info] << "IO::~IO()";}
	bool step() {
		log[log::info] << "IO::step()";
		core::pFrame f = pop_frame(0);
		if (!f) {
			sleep(0.2_s);

		} else {
			request_end(core::yuri_exit_finished);
		}
		return true;

	}
//	void run() {log[log::info] << "IO::run()";}
};
IOTHREAD_GENERATOR(IO)

REGISTER_IOTHREAD("io", IO)

class app: public core::ThreadBase
{
public:
	app(const log::Log& log_):ThreadBase(log_,core::pwThreadBase()) {
		log.set_label("[APP] ");
		log[log::info] << "app::app()";
	}
	~app() noexcept {log[log::info] << "app::~app()";}
	void run() override {
		log[log::info] << "app::run()";
		core::Parameters pparams = core::PipeGenerator::get_instance().configure("single_blocking");
		core::pPipe pajp = core::PipeGenerator::get_instance().generate("single_blocking","pajpa", log, std::move(pparams));
//		core::pIOThread io = make_shared<IO>(log, get_this_ptr(), core::Parameters());
//		core::Parameters params = IO::configure();
		core::Parameters params = IOThreadGenerator::get_instance().configure("io");
		params["name"] = "xxx";
		core::Parameters pp;
		pp["abcd"]="franta";
//		params.merge(std::move(pp));
		params.merge(pp);

//		core::pIOThread io = IO::generate(log, get_this_ptr(), params);
		core::pIOThread io = IOThreadGenerator::get_instance().generate("io",log, get_this_ptr(), params);
		io->connect_in(0, pajp);
		log[log::info] << "app::run() spawning thread";
		spawn_thread(io);
		log[log::info] << "app::run() spawned thread";
		core::pFrame f = yuri::make_shared<core::RawVideoFrame>(0, resolution_t{640,480});
		sleep(2_s);
		pajp->push_frame(f);
		while (still_running()) {
			sleep(100_ms);
		}
	}
private:
};



int main(int argc, char** argv)
{
	l.set_label("[CORE TEST] ");
	l[log::info] << "Hey";
	for (const auto& pl: core::PipeGenerator::get_instance().list_keys()) {
		l[log::info] << "Trying to create a pipe of type " << pl;
		try {
			core::pPipe p = core::PipeGenerator::get_instance().generate(pl,"pajpa",l,core::Parameters());
			l[log::info] << "Pipe type " << pl << " successfully created!";
		}
		catch (std::runtime_error&) {
			l[log::error] << "Pipe type " << pl << " failed!";
		}
	}
	core::pPipe p = core::PipeGenerator::get_instance().generate("single_blocking", "pajpa", l, core::Parameters());
//	core::pFrame f = yuri::make_shared<core::RawVideoFrame>(0, resolution_t{640,480});

	size_t check_size = 640*480*3;
	l[log::info] << "Pre-allocated frames of size " << check_size << "B: " << core::FixedMemoryAllocator::preallocated_blocks(check_size);
	core::pRawVideoFrame f = core::RawVideoFrame::create_empty(core::raw_format::rgb24,{640,480}, true);
	l[log::info] << "Frame of with resolution " << f->get_resolution() << " has size " << f->get_size() << " bytes";
	l[log::info] << "Pre-allocated frames of size " << check_size << "B: " << core::FixedMemoryAllocator::preallocated_blocks(check_size);
	if (!p->push_frame(f->get_copy())) l[log::error] << "Failed to push frame into the pipe";
	if (!p->push_frame(f->get_copy())) l[log::error] << "Failed to push second frame into the pipe (EXPECTED)";
	core::pFrame f2 = p->pop_frame();
	if (f2) {
		l[log::info] << "Got frame back from pipe";
	}
	f2.reset();
	f.reset();
	l[log::info] << "Pre-allocated frames of size " << check_size << "B: " << core::FixedMemoryAllocator::preallocated_blocks(check_size);
	l.set_flags(log::verbose_debug);
	core::pThreadBase a = make_shared<app>(l);
	l[log::info] << "Instances of a: " << a.use_count();
	(*a)();

	core::Parameter par("str",998);
	par = true;
	l[log::info] << "Param has value: " <<par.get<std::string>();
	par = "Nazdar";
	l[log::info] << "Param has value: " <<par.get<std::string>();
	par = std::string("Nazdar 2");
	l[log::info] << "Param has value: " <<par.get<std::string>();
	par = 37;
	l[log::info] << "Param has value: " <<par.get<std::string>();
	par = 4.597;
	l[log::info] << "Param has value: " <<par.get<std::string>();
	par["hey hey hey"]=5;
	l[log::info] << "Param has value: " <<par.get<std::string>();

	par = "nazdar";
	core::Parameter par2("x");
	par2 = par;
	l[log::info] << "(copy)\nParam has value: " <<par.get<std::string>();
	l[log::info] << "Param2 has value: " <<par2.get<std::string>();
	par2 = std::move(par);
//	l[log::info] << "(move)\nParam has value: " <<par.get<std::string>();
	l[log::info] << "Param2 has value: " <<par2.get<std::string>();

	core::Parameters params;
	params["xxx"]["aaa"]=8.75;
	l[log::info] << "Param[xxx] has value: " <<params["xxx"].get<std::string>();
	l[log::info] << "Param[xxx] has value (as float): " <<params["xxx"].get<float>();
	l[log::info] << "Param[xxx] has value (as int): " <<params["xxx"].get<int>();



	par = resolution_t{640,480};
	l[log::info] << "Param has value: " <<par.get<std::string>();
	l[log::info] << "Param has value (as resolution_t): " <<par.get<resolution_t>();

	par = "1920x1080+500+-300";
	l[log::info] << "Param has value: " <<par.get<std::string>();
	l[log::info] << "Param has value (as geometry_t): " <<par.get<geometry_t>();

	format_t fmt = core::raw_format::parse_format("YUV");
	if (fmt == core::raw_format::unknown) {
		l[log::info] << "Failed to parse format YUV";
	} else {
		const auto& finfo = core::raw_format::get_format_info(fmt);
		l[log::info] << "Parsed format YUV to: " << finfo.name;
	}
	for (const auto& xx: core::module_loader::get_builtin_paths()) {
		for (const auto& path: core::module_loader::find_modules_path(xx)) {
			l[log::info] << "Loading " << path << ": " << std::boolalpha <<
					core::module_loader::load_module(path);
		}

	}
//	l[log::info] << "Loading null: " << std::boolalpha << core::module_loader::load_module("./bin/modules/yuri2.8_module_null.so");
//	l[log::info] << "Loading dup: " << std::boolalpha << core::module_loader::load_module("./bin/modules/yuri2.8_module_dup.so");
	const auto& keys = IOThreadGenerator::get_instance().list_keys();
	for (const auto& k: keys) {
		l[log::info] << "Registered io thread: " << k;
	}
	core::Parameters fffp = IOThreadGenerator::get_instance().configure("null");
	auto fff = IOThreadGenerator::get_instance().generate("null",l,core::pwThreadBase(), fffp);

	std::vector<std::string> vargv;
	for (int i = 1; i< argc; ++i) {
		vargv.push_back(argv[i]);
	}
	core::XmlBuilder builder{l, core::pwThreadBase{}, "single_webcam.xml", vargv};
}

