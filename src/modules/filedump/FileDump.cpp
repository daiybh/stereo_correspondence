/*!
 * @file 		FileDump.cpp
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "FileDump.h"
#include <sstream>
#include <iomanip>
#include "yuri/core/Module.h"
namespace yuri
{
namespace dump
{

REGISTER("filedump",FileDump)

core::pBasicIOThread FileDump::generate(log::Log &_log,core::pwThreadBase parent, core::Parameters& parameters)
{
	shared_ptr<FileDump> dump;
	try {
		dump.reset(new FileDump(_log,parent,parameters));
	}
	catch (std::exception &e) {
		throw exception::InitializationFailed(std::string("Filedump constuctor failed: ") + e.what());
	}
	return dump;
}
core::pParameters FileDump::configure()
{
	core::pParameters p (new core::Parameters());
	p->set_description("Outputs frames to a file. It either dump all frames to one file of every frame to separate file. ");
	(*p)["sequence"]["Number of digits in sequence number. Set to 0 to disable sequence and output all frames to one file."]=0;
	(*p)["filename"]["Required parameter. Path of file to dump to"]=std::string();
	(*p)["frame_limit"]["Maximal number of frames to dump. 0 for unlimited"]=0;
	return p;
}

FileDump::FileDump(log::Log &log_,core::pwThreadBase parent,core::Parameters &parameters):
	BasicIOThread(log_,parent,1,0,"Dump"),
	dump_file(),filename(),seq_chars(0),seq_number(0),dumped_frames(0),
	dump_limit(0)
{
	params.merge(*configure());
	params.merge(parameters);
	filename = params["filename"].get<std::string>();
	seq_chars = params["sequence"].get<int>();
	dump_limit = params["frame_limit"].get<yuri::size_t>();
	if (!seq_chars)
		dump_file.open(filename.c_str(), std::ios::binary | std::ios::out);
}

FileDump::~FileDump()
{
	if (dump_file.is_open()) dump_file.close();
}

bool FileDump::step()
{
	if (!in[0]) return true;
	core::pBasicFrame f;
	while ((f = in[0]->pop_frame()).get()) {
		if (seq_chars) {
			std::stringstream ss;
			ss << filename.substr(0,filename.find_last_of('.'))
				<< std::setfill('0') << std::setw(seq_chars) << seq_number++
				<< filename.substr(filename.find_last_of('.'));
			dump_file.open(ss.str().c_str(), std::ios::binary | std::ios::out);
		}
		log[log::debug]<<"Dumping " << f->get_planes_count() << " planes" << std::endl;
		for (yuri::size_t i=0; i<f->get_planes_count();++i) {
			//log[log::debug]<<"Dumping plane " << i << ", size: " << (*f)[i].get_size() << std::endl;
//			dump_file.write((const char *)(*f)[i].data.get(),(*f)[i].size);
			dump_file.write(reinterpret_cast<const char *>(PLANE_RAW_DATA(f,0)),PLANE_SIZE(f,0));
		}
		if (seq_chars) {
			dump_file.close();
		}
		// The comparison is evaluated FIRST in order to have dumped_frames counted even if dump_limit is zero
		if (++dumped_frames >= dump_limit && dump_limit) {
			log[log::debug] << "Maximal number of frames reached, quitting." << std::endl;
			exitCode = YURI_EXIT_FINISHED;
			request_end();
			break;
		}
	}
	return true;
}

}
}
//End of File

