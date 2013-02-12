/*
 * Fetcher.cpp
 *
 *  Created on: Jul 28, 2009
 *      Author: neneko
 */

#include "Fetcher.h"

namespace yuri {

namespace io {

REGISTER("curlfetcher",Fetcher)

shared_ptr<BasicIOThread> Fetcher::generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception)
{
	shared_ptr<Fetcher> fetch (new Fetcher(_log,parent,
			parameters["url"].get<string>()));
	fetch->setUploadParams(parameters["filename"].get<string>(),
			parameters["filetype"].get<string>(),
			parameters["inputname"].get<string>());
	return fetch;
}
shared_ptr<Parameters> Fetcher::configure()
{
	shared_ptr<Parameters> p (new Parameters());
	(*p)["url"]=string();
	(*p)["filename"]=string();
	(*p)["filetype"]=string();
	(*p)["inputname"]=string();
	return p;
}


Fetcher::Fetcher(Log &_log, pThreadBase parent, string url) throw (Exception)
	:BasicIOThread(_log,parent,1,1,"Fetcher"),url(url),
	temp_data(ios::in|ios::out|ios::binary)
{
	curl_global_init(CURL_GLOBAL_ALL);
	curl.reset(curl_easy_init(),curl_session_deleter);
	if (!curl) throw Exception("Can't initialize curl!");
}

Fetcher::~Fetcher()
{

}

shared_ptr<BasicFrame> Fetcher::fetch()
{
	assert(curl);
	shared_ptr<BasicFrame> f, f_dummy;
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
		boost::mutex::scoped_lock(upload_lock);
		log[debug] << "Uploading file " << fname << " with size "
			<< f->get_size() << endl;

		if((cerror = curl_formadd(&first,&last,
				CURLFORM_COPYNAME,iname.c_str(),
				CURLFORM_CONTENTTYPE,ftype.c_str(),
				CURLFORM_BUFFER,fname.c_str(),
				CURLFORM_BUFFERPTR,(*f)[0].data.get(),
				CURLFORM_BUFFERLENGTH,(*f)[0].size,
				CURLFORM_END))) {
			log[warning] << "Failed to pack file into httppost structure! "
				<< "Error code " << printCurlFormaddError(cerror) << endl;

			//delete f;
			//f = 0;
			f.reset();
			curl_formfree(first);
			first = 0;
			last = 0;
		}
	}
	if (!sections.empty()) for (map<string,string>::iterator it = sections.begin();
		it != sections.end(); ++it) {
		if ((cerror=curl_formadd(&first,&last,
				CURLFORM_COPYNAME,(*it).first.c_str(),
				CURLFORM_COPYCONTENTS,(*it).second.c_str(),
				CURLFORM_END))) {
			log[warning] << "Failed to pack section " << (*it).first
				<< "into httppost structure! Error: "
				<< printCurlFormaddError(cerror) << endl;
		}
	}
	if (first) curl_easy_setopt(curl.get(),CURLOPT_HTTPPOST,first);

	log[debug] << "Fetching " << url << std::endl;
	if (curl_easy_perform(curl.get())) {
		log[debug] << "failed!!" << endl;
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
		BasicIOThread::sleep(latency);
	}
}

bool Fetcher::step()
{
	shared_ptr<BasicFrame> f = fetch();
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
	log[verbose_debug] << "Writing " << length << " bytes to stringstream" << endl;
	temp_data.write((const char*)data,length);
	if (temp_data.bad()) return 0;
	return length;
}

shared_ptr<BasicFrame> Fetcher::dumpData()
{
	yuri::size_t length = temp_data.tellp();
	shared_ptr<BasicFrame> frame (new BasicFrame(1));
	shared_array<yuri::ubyte_t> mem = allocate_memory_block(length+1);
	(*frame)[0].set(mem,length+1);
	log[debug] << "Reading " << length << " bytes from stringstream" << endl;
	//char *mem = new char[length+1];
	temp_data.seekg(0,ios::beg);
	temp_data.read(reinterpret_cast<char*>(mem.get()),length);
	temp_data.seekp(0,ios::beg);
	temp_data.str().clear();
	mem[length]=0;
	/*if (response && osize) {
		*response=mem;
		*osize = length;
		return 0;
	}*/

	const char *c_ptr = 0;
	curl_easy_getinfo(curl.get(),CURLINFO_CONTENT_TYPE,&c_ptr);
	log[debug] << "Fetched media " << c_ptr << endl;
	type = BasicPipe::set_frame_from_mime(frame,c_ptr);
	log[debug] << "Format set to " << BasicPipe::get_format_string(frame->get_format()) << endl;
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

void Fetcher::setUploadParams(string filename, string filetype, string inputname)
{
	boost::mutex::scoped_lock(upload_lock);
	fname = filename;
	ftype = filetype;
	iname = inputname;
}

void Fetcher::addUploadSection(string name, string value)
{
	boost::mutex::scoped_lock(upload_lock);
	sections[name]=value;
}

string Fetcher::printCurlFormaddError(int cerror)
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
	boost::mutex::scoped_lock(upload_lock);
	sections.clear();
}

void Fetcher::setUrl(string new_url)
{
	boost::mutex::scoped_lock(upload_lock);
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

