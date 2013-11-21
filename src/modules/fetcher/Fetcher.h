/*!
 * @file 		Fetcher.h
 * @author 		Zdenek Travnicek
 * @date 		28.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2009 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
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
#include "yuri/core/IOThread.h"

namespace yuri {

namespace fetcher {

class Fetcher: public core::IOThread {
public:
	Fetcher(log::Log &_log, core::pwThreadBase parent, std::string url);
	virtual ~Fetcher();
	static core::pIOThread generate(log::Log &_log,core::pwThreadBase parent, core::Parameters& parameters);
	static core::pParameters configure();

	core::pBasicFrame fetch();
	virtual void run();
	void setUploadParams(std::string filename, std::string filetype, std::string inputname);
	void addUploadSection(std::string name, std::string value);
	void clearUploadSections();
	void setUrl(std::string new_url);
	//static void deleter_curl_form(struct curl_httppost *h);
	static void curl_session_deleter(CURL *pcurl);
protected:
	static yuri::size_t writeCallback(void* data, size_t size, size_t num, void*stream);
	virtual bool step();
	yuri::size_t writeData(void* data, size_t size, size_t num);
	core::pBasicFrame dumpData();
	std::string printCurlFormaddError(int cerror);
	std::string url;
	shared_ptr<CURL> curl;
	std::stringstream temp_data;
	std::string fname, ftype, iname;
	std::map<std::string, std::string> sections;
	yuri::mutex upload_lock;
	long type;
};

}

}

#endif /* FETCHER_H_ */
