/*!
 * @file 		FileDump.cpp
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "FileDump.h"
#include <sstream>
#include <iomanip>
#include "yuri/core/Module.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
namespace yuri
{
namespace dump
{

MODULE_REGISTRATION_BEGIN("filedump")
	REGISTER_IOTHREAD("filedump",FileDump)
MODULE_REGISTRATION_END()

IOTHREAD_GENERATOR(FileDump)

core::Parameters FileDump::configure()
{
	core::Parameters p = IOFilter::configure();
	p.set_description("Outputs frames to a file. It either dump all frames to one file of every frame to separate file. ");
	p["sequence"]["Number of digits in sequence number. Set to 0 to disable sequence and output all frames to one file."]=0;
	p["filename"]["Required parameter. Path of file to dump to"]=std::string();
	p["frame_limit"]["Maximal number of frames to dump. 0 for unlimited"]=0;
	return p;
}

FileDump::FileDump(log::Log &log_,core::pwThreadBase parent, const core::Parameters &parameters):
	IOFilter(log_,parent,"Dump"),
	dump_file(),filename(),seq_chars(0),seq_number(0),dumped_frames(0),
	dump_limit(0)
{
	IOTHREAD_INIT(parameters);
	if (filename.empty()) throw exception::InitializationFailed("No filename specified");
	if (!seq_chars)
		dump_file.open(filename.c_str(), std::ios::binary | std::ios::out);
}

FileDump::~FileDump() noexcept
{
	if (dump_file.is_open()) dump_file.close();
}

core::pFrame FileDump::do_simple_single_step(const core::pFrame& frame)
{
	if (seq_chars) {
		std::stringstream ss;
		const auto dot_index = filename.find_last_of('.');
		if (dot_index != std::string::npos) {
			ss << filename.substr(0,filename.find_last_of('.'));
		} else {
			ss << filename;
		}
		ss << std::setfill('0') << std::setw(seq_chars) << seq_number++;
		if (dot_index != std::string::npos) {
			ss << filename.substr(filename.find_last_of('.'));
		}
		log[log::info] << "FIlename: " << ss.str() << ", idx " << dot_index;
		dump_file.open(ss.str().c_str(), std::ios::binary | std::ios::out);
	}
	if (auto f = std::dynamic_pointer_cast<core::RawVideoFrame>(frame)) {
		log[log::debug]<<"Dumping " << f->get_planes_count() << " planes";
		for (yuri::size_t i=0; i<f->get_planes_count();++i) {
			dump_file.write(reinterpret_cast<const char *>(PLANE_RAW_DATA(f,0)),PLANE_SIZE(f,0));
		}
	} else if (auto f = std::dynamic_pointer_cast<core::CompressedVideoFrame>(frame)) {
		dump_file.write(reinterpret_cast<const char *>(f->begin()),f->size());
	}
	if (seq_chars) {
		dump_file.close();
	}
	// The comparison is evaluated FIRST in order to have dumped_frames counted even if dump_limit is zero
	if (++dumped_frames >= dump_limit && dump_limit) {
		log[log::debug] << "Maximal number of frames reached, quitting.";
		request_end();
	}

	return {};
}

bool FileDump::set_param(const core::Parameter &param)
{
	if (param.get_name() == "filename") {
		filename = param.get<std::string>();
	} else if (param.get_name() == "sequence") {
		seq_chars = param.get<int>();
	} else if (param.get_name() == "frame_limit") {
		dump_limit = param.get<size_t>();
	} else return IOFilter::set_param(param);
	return true;
}

}
}
//End of File

