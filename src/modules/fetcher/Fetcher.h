/*!
 * @file 		Fetcher.h
 * @author 		Zdenek Travnicek
 * @date 		28.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 *	Fetcher provides very simplified access to curl-based file fetching
 *	When there's only output pipe provided, it simply fetches url once
 *	every <latency> us adn pushes it into pipe.
 *	When there's also an input pipe, it fetches frames from input pipe and
 *	tries to upload them to the url.
 */

#ifndef FETCHER_H_
#define FETCHER_H_
#include <curl/curl.h>
#include <sstream>
#include <map>
#include "yuri/io/BasicIOThread.h"
#include "yuri/exception/Exception.h"
#include "yuri/config/RegisteredClass.h"

namespace yuri {

namespace io {
using yuri::log::Log;
using namespace std;
using yuri::exception::Exception;
using namespace yuri::config;

class Fetcher: public BasicIOThread {
public:
	Fetcher(Log &_log, pThreadBase parent, string url) throw (Exception);
	virtual ~Fetcher();
	static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception);
	static shared_ptr<Parameters> configure();

	shared_ptr<BasicFrame> fetch();
	virtual void run();
	void setUploadParams(string filename, string filetype, string inputname);
	void addUploadSection(string name, string value);
	void clearUploadSections();
	void setUrl(string new_url);
	//static void deleter_curl_form(struct curl_httppost *h);
	static void curl_session_deleter(CURL *pcurl);
protected:
	static yuri::size_t writeCallback(void* data, size_t size, size_t num, void*stream);
	virtual bool step();
	yuri::size_t writeData(void* data, size_t size, size_t num);
	shared_ptr<BasicFrame> dumpData();
	string printCurlFormaddError(int cerror);
	string url;
	shared_ptr<CURL> curl;
	stringstream temp_data;
	string fname, ftype, iname;
	map<string, string> sections;
	boost::mutex upload_lock;
	long type;
};

}

}

#endif /* FETCHER_H_ */
