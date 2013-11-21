/*!
 * @file 		Fetcher.cpp
 * @author 		Zdenek Travnicek
 * @date 		28.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2009 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Fetcher.h"
#include "yuri/core/Module.h"
namespace yuri {

namespace fetcher {

REGISTER("curlfetcher",Fetcher)

core::pIOThread Fetcher::generate(log::Log &_log, core::pwThreadBase parent, core::Parameters& parameters)
{
	shared_ptr<Fetcher> fetch (new Fetcher(_log,parent,
			parameters["url"].get<std::string>()));
	fetch->setUploadParams(parameters["filename"].get<std::string>(),
			parameters["filetype"].get<std::string>(),
			parameters["inputname"].get<std::string>());
	return fetch;
}
core::pParameters Fetcher::configure()
{
	core::pParameters p (new core::Parameters());
	(*p)["url"]=std::string();
	(*p)["filename"]=std::string();
	(*p)["filetype"]=std::string();
	(*p)["inputname"]=std::string();
	return p;
}


Fetcher::Fetcher(log::Log &_log, core::pwThreadBase parent, std::string url)
	:core::IOThread(_log,parent,1,1,"Fetcher"),url(url),
	temp_data(std::ios::in|std::ios::out|std::ios::binary)
{
	curl_global_init(CURL_GLOBAL_ALL);
	curl.reset(curl_easy_init(),curl_session_deleter);
	if (!curl) throw exception::Exception("Can't initialize curl!");
}

Fetcher::~Fetcher()
{

}

core::pBasicFrame Fetcher::fetch()
{
	assert(curl);
	core::pBasicFrame f, f_dummy;
	struct curl_httppost *first=0,*last=0;
	int cerror = 0;
	if (in[0]) {
		f = in[0]->pop_frame();
		if (!f) return f_dummy;
	}
	curl_easy_setopt(curl.get(),CURLOPT_NOPROGRESS,1);
	curl_easy_setopt(curl.get(),CURLOPT_WRITEFUNCTION,Fetcher::writeCallback);
	curl_easy_setopt(curl.get(),CURLOPT_WRITEDATA,(void*)this);
	curl_easy_setopt(curl.get(),CURLOPT_URL,url.c_str());
	curl_easy_setopt(curl.get(),CURLOPT_USERAGENT,"Neneko @ libyuri");
	if (f) { // Uploading a file
		yuri::lock_t(upload_lock);
		log[log::debug] << "Uploading file " << fname << " with size "
			<< f->get_size();

		if((cerror = curl_formadd(&first,&last,
				CURLFORM_COPYNAME,iname.c_str(),
				CURLFORM_CONTENTTYPE,ftype.c_str(),
				CURLFORM_BUFFER,fname.c_str(),
				CURLFORM_BUFFERPTR,PLANE_RAW_DATA(f,0),
				CURLFORM_BUFFERLENGTH,PLANE_SIZE(f,0),
				CURLFORM_END))) {
			log[log::warning] << "Failed to pack file into httppost structure! "
				<< "Error code " << printCurlFormaddError(cerror) << std::endl;

			//delete f;
			//f = 0;
			f.reset();
			curl_formfree(first);
			first = 0;
			last = 0;
		}
	}
	if (!sections.empty()) for (std::map<std::string,std::string>::iterator it = sections.begin();
		it != sections.end(); ++it) {
		if ((cerror=curl_formadd(&first,&last,
				CURLFORM_COPYNAME,(*it).first.c_str(),
				CURLFORM_COPYCONTENTS,(*it).second.c_str(),
				CURLFORM_END))) {
			log[log::warning] << "Failed to pack section " << (*it).first
				<< "into httppost structure! Error: "
				<< printCurlFormaddError(cerror) << std::endl;
		}
	}
	if (first) curl_easy_setopt(curl.get(),CURLOPT_HTTPPOST,first);

	log[log::debug] << "Fetching " << url << std::endl;
	if (curl_easy_perform(curl.get())) {
		log[log::debug] << "failed!!" << std::endl;
		if (first) curl_formfree(first);
		return f_dummy;
	}
	if (first) curl_formfree(first);
	first = 0; last = 0;

	return dumpData();
}

void Fetcher::run()
{
	while(still_running()) {
		if (!step()) break;
		IOThread::sleep(latency);
	}
}

bool Fetcher::step()
{
	core::pBasicFrame f = fetch();
	if (out[0] && f) {
		out[0]->set_type(type);
		push_raw_frame(0,f);
	}
	return true;

}

yuri::size_t Fetcher::writeCallback(void* data, size_t size, size_t num, void*stream)
{
	if (!stream) return 0;
	Fetcher *f = (Fetcher*)stream;
	return f->writeData(data,size,num);
}

yuri::size_t Fetcher::writeData(void* data, size_t size, size_t num)
{
	yuri::size_t length = size*num;
	log[log::verbose_debug] << "Writing " << length << " bytes to std::stringstream" << std::endl;
	temp_data.write((const char*)data,length);
	if (temp_data.bad()) return 0;
	return length;
}

core::pBasicFrame Fetcher::dumpData()
{
	yuri::size_t length = temp_data.tellp();
	core::pBasicFrame frame (new core::BasicFrame(1));
	//shared_array<yuri::ubyte_t> mem = allocate_memory_block(length+1);
	//(*frame)[0].set(mem,length+1);
//	PLANE_DATA(frame,0).resize(length+1);
	log[log::debug] << "Reading " << length << " bytes from std::stringstream" << std::endl;
	//char *mem = new char[length+1];
	temp_data.seekg(0,std::ios::beg);
//	temp_data.read(reinterpret_cast<char*>(mem.get()),length);
	const std::string& str = temp_data.str();
	frame->set_plane(0,reinterpret_cast<const yuri::ubyte_t *>(str.c_str()),str.size()+1);
	temp_data.seekp(0,std::ios::beg);
	temp_data.str().clear();
//	PLANE_DATA(frame,0).push_back(0);
//	mem[length]=0;
	/*if (response && osize) {
		*response=mem;
		*osize = length;
		return 0;
	}*/

	const char *c_ptr = 0;
	curl_easy_getinfo(curl.get(),CURLINFO_CONTENT_TYPE,&c_ptr);
	log[log::debug] << "Fetched media " << c_ptr << std::endl;
	type = core::BasicPipe::set_frame_from_mime(frame,c_ptr);
	log[log::debug] << "Format set to " << core::BasicPipe::get_format_string(frame->get_format()) << std::endl;
	return frame;
	/*
	if (!out[0]) {
		delete [] mem;
		return 0;
	}
	else {
		out[0]->push_frame(mem,length,true);
	}
	return length;*/
}

void Fetcher::setUploadParams(std::string filename, std::string filetype, std::string inputname)
{
	yuri::lock_t(upload_lock);
	fname = filename;
	ftype = filetype;
	iname = inputname;
}

void Fetcher::addUploadSection(std::string name, std::string value)
{
	yuri::lock_t(upload_lock);
	sections[name]=value;
}

std::string Fetcher::printCurlFormaddError(int cerror)
{
	switch (cerror) {
	case CURL_FORMADD_OK: return "CURL_FORMADD_OK";
	case CURL_FORMADD_MEMORY: return "CURL_FORMADD_MEMORY";
	case CURL_FORMADD_OPTION_TWICE: return "CURL_FORMADD_OPTION_TWICE";
	case CURL_FORMADD_NULL: return "CURL_FORMADD_NULL";
	case CURL_FORMADD_UNKNOWN_OPTION: return "CURL_FORMADD_UNKNOWN_OPTION";
	case CURL_FORMADD_INCOMPLETE: return "CURL_FORMADD_INCOMPLETE";
	case CURL_FORMADD_ILLEGAL_ARRAY: return "CURL_FORMADD_ILLEGAL_ARRAY";
	case CURL_FORMADD_DISABLED: return "CURL_FORMADD_DISABLED";
	}
	return "Unknown error";
}
void Fetcher::clearUploadSections()
{
	yuri::lock_t(upload_lock);
	sections.clear();
}

void Fetcher::setUrl(std::string new_url)
{
	yuri::lock_t(upload_lock);
	url=new_url;
}
/*
static void deleter_curl_form(struct curl_httppost *h)
{
	curl_formfree(h);
}*/

void Fetcher::curl_session_deleter(CURL *pcurl)
{
	curl_easy_cleanup(pcurl);
}
}

}

