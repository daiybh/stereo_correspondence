/*!
 * @file 		RawFileSource.cpp
 * @author 		Zdenek Travnicek
 * @date 		31.3.2011
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2011 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "RawFileSource.h"
#include <fstream>
#include <boost/assign.hpp>
namespace yuri {

namespace io {


REGISTER("raw_filesource",RawFileSource)

shared_ptr<BasicIOThread> RawFileSource::generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception)
{
	shared_ptr<BasicIOThread> c (new RawFileSource(_log,parent,parameters));
	return c;
}

shared_ptr<Parameters> RawFileSource::configure()
{
	shared_ptr<Parameters> p = BasicIOThread::configure();
	(*p)["path"]["Path to the file"]=std::string();
	(*p)["keep_alive"]["Stay idle after pushing the file (setting to false will cause the object to quit afterward)"]=true;
	(*p)["output_format"]["Force output format"]=std::string("single16");
	(*p)["width"]["Force output width to"]=0;
	(*p)["height"]["Force output height to"]=0;
	(*p)["chunk"]["Chunk size (0 to output whole file at once)"]=0;
	(*p)["fps"]["Framerate for chunk output"]=25;
	(*p)["loop"]["Start againg from beginning of the file after reaching end"]=true;
	(*p)["offset"]["skip offset bytes from beginning"]=0;
	(*p)["block"]["Threat output pipes as blocking. Specify as max number of frames in output pipe."]=0;
	p->set_max_pipes(0,1);
	p->add_output_format(YURI_FMT_NONE);
	return p;
}


RawFileSource::RawFileSource(Log &_log, pThreadBase parent, Parameters &parameters):
			BasicIOThread(_log,parent,1,2,"RawFileSource"), position(0),
			chunk_size(0), width(0), height(0),output_format(YURI_FMT_NONE),
			fps(25.0),last_send(not_a_date_time),keep_alive(true),loop(true),
			failed_read(false),block(0),loop_number(0)
{
	IO_THREAD_INIT("Raw file source")
	latency = 1000;
}

RawFileSource::~RawFileSource() {
}

void RawFileSource::run()
{
	IO_THREAD_PRE_RUN
	while (still_running()) {
		ThreadBase::sleep(latency);
		if (!frame) if (!read_chunk()) break;
		if (failed_read) break;
		if (!frame) continue;
		if (block && out[0] && out[0]->get_count() >= block) continue;
		if (last_send != not_a_date_time) {
			time_duration delta;
			if (fps!=0.0)
				delta = microseconds(1e6/fps);
			else delta = microseconds(0);
			if ((microsec_clock::local_time()-last_send) < delta) continue;
			last_send+=delta;
		} else {
			last_send=microsec_clock::local_time();
		}
		if (out[0]) {
			push_raw_frame(0,frame);
			if (chunk_size) frame.reset();
			if (!loop && loop_number) break;
		}
	}
	if (keep_alive) while (still_running()) {
		ThreadBase::sleep(latency);
	}
	IO_THREAD_POST_RUN
}

bool RawFileSource::read_chunk()
{
	try {
		bool first_read = false;
		if (!file.is_open()) {
			file.open(path.c_str(),std::ifstream::in);
			file.seekg(position,std::ios_base::beg);
			first_read = true;
		}
		yuri::size_t length = chunk_size;
		std::vector<yuri::size_t> planes=boost::assign::list_of(length);
		FormatInfo_t fi = BasicPipe::get_format_info(output_format);
		if (output_format != YURI_FMT_NONE && fi && !fi->compressed && width && height) {

			// Known raw format. I can guess the chunk_size.
			length = (width*height*fi->bpp)>>3;
			if (fi->planes>1)
			{
				planes.clear();
				for (yuri::size_t i=0;i<fi->planes;++i)
				{
					planes.push_back((width*height*fi->component_depths[i]/fi->plane_x_subs[i]/fi->plane_y_subs[i])>>3);
				}

			} else planes=boost::assign::list_of(length);
			log[debug] << "Guessed size for " << fi->long_name << " is " << length<<std::endl;
		} else if (!chunk_size) {
			file.seekg(0,std::ios_base::end);
			length = static_cast<yuri::size_t>(file.tellg()) - position;
			file.seekg(position,std::ios_base::beg);
			planes=boost::assign::list_of(length);
		}
		frame.reset(new BasicFrame(planes.size()));
		//(*frame)[0].set(data,length);
		for (yuri::size_t i=0;i<planes.size();++i) {
			yuri::size_t plane_length = planes[i];
			std::istreambuf_iterator<char> it(file);
			frame->get_plane(i).resize(plane_length);
//			std::copy(it,it+plane_length,std::back_inserter(frame->get_plane(0)));
//			shared_array<yuri::ubyte_t> data = BasicIOThread::allocate_memory_block(plane_length,true);

			file.read(reinterpret_cast<char*>(PLANE_RAW_DATA(frame,i)),plane_length);
			if (static_cast<yuri::size_t>(file.gcount()) != plane_length ) {
				if (first_read) {
					failed_read=true;
					log[warning]<< "Wrong length of the file (read " << file.gcount() << ", expected " << plane_length << ")" << std::endl;
				}
				file.close();
				frame.reset();++loop_number;
				return !failed_read;
			}
//			(*frame)[i].set(data,plane_length);
		}
		if (file.eof()) {
			file.close();
			++loop_number;
		}
		//frame.reset(new BasicFrame(1));
		//(*frame)[0].set(data,length);
		frame->set_parameters(output_format, width, height);
	}
	catch (std::exception &e) {
		log[error] << "Failed to process file " << params["path"].get<std::string>()
				<< " (" << e.what() << ")"<<std::endl;
		failed_read=true;
		return false;
	}
	return true;
}
bool RawFileSource::set_param(Parameter &parameter)
{
	if (parameter.name == "chunk") {
		chunk_size=parameter.get<yuri::size_t>();
	} else if (parameter.name == "fps") {
		fps=parameter.get<double>();
	} else if (parameter.name == "width") {
		width=parameter.get<yuri::size_t>();
	} else if (parameter.name == "height") {
		height=parameter.get<yuri::size_t>();
	} else if (parameter.name == "output_format") {
		output_format = BasicPipe::get_format_from_string(parameter.get<std::string>());
		log[info] << "output format " << output_format << std::endl;
	} else if (parameter.name == "path") {
		path=parameter.get<std::string>();
	} else if (parameter.name == "keep_alive") {
		keep_alive=parameter.get<bool>();
	} else if (parameter.name == "offset") {
		position=parameter.get<yuri::size_t>();
	} else if (parameter.name == "loop") {
		loop=parameter.get<bool>();
	} else if (parameter.name == "block") {
		block=parameter.get<yuri::size_t>();
	} else return BasicIOThread::set_param(parameter);
	return true;
}


}
}
