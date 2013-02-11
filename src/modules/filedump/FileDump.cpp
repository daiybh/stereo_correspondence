#include "FileDump.h"
#include <sstream>
#include "yuri/config/RegisteredClass.h"
namespace yuri
{
namespace io
{

REGISTER("filedump",FileDump)

shared_ptr<BasicIOThread> FileDump::generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception)
{
	shared_ptr<FileDump> dump;
	try {
		dump.reset(new FileDump(_log,parent,parameters));
	}
	catch (std::exception &e) {
		throw InitializationFailed(std::string("Filedump constuctor failed: ") + e.what());
	}
	return dump;
}
shared_ptr<Parameters> FileDump::configure()
{
	shared_ptr<Parameters> p (new Parameters());
	p->set_description("Outputs frames to a file. It either dump all frames to one file of every frame to separate file. ");
	(*p)["sequence"]["Number of digits in sequence number. Set to 0 to disable sequence and output all frames to one file."]=0;
	(*p)["filename"]["Required parameter. Path of file to dump to"]=std::string();
	(*p)["frame_limit"]["Maximal number of frames to dump. 0 for unlimited"]=0;
	return p;
}

FileDump::FileDump(Log &log_,pThreadBase parent, Parameters &parameters):
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
	pBasicFrame f;
	while ((f = in[0]->pop_frame()).get()) {
		if (seq_chars) {
			std::stringstream ss;
			ss << filename.substr(0,filename.find_last_of('.'))
				<< std::setfill('0') << std::setw(seq_chars) << seq_number++
				<< filename.substr(filename.find_last_of('.'));
			dump_file.open(ss.str().c_str(), std::ios::binary | std::ios::out);
		}
		log[debug]<<"Dumping " << f->get_planes_count() << " planes" << std::endl;
		for (yuri::size_t i=0; i<f->get_planes_count();++i) {
			//log[debug]<<"Dumping plane " << i << ", size: " << (*f)[i].get_size() << std::endl;
			dump_file.write((const char *)(*f)[i].data.get(),(*f)[i].size);
		}
		if (seq_chars) {
			dump_file.close();
		}
		// The comparison is evaluated FIRST in order to have dumped_frames counted even if dump_limit is zero
		if (++dumped_frames >= dump_limit && dump_limit) {
			log[debug] << "Maximal number of frames reached, quitting." << std::endl;
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

