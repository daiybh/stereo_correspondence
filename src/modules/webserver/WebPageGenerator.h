/*
 * WebPageGenerator.h
 *
 *  Created on: Dec 7, 2014
 *      Author: neneko
 */

#ifndef SRC_MODULES_WEBSERVER_WEBPAGEGENERATOR_H_
#define SRC_MODULES_WEBSERVER_WEBPAGEGENERATOR_H_
#include "common_types.h"
#include <vector>
#include <string>
namespace yuri {
namespace webserver {

struct web_meta_t {
	std::string name;
	std::string content;
};

struct webpage_t {
	std::string title;
	std::vector<web_meta_t> metas;
	std::string body;
	bool default_footer;
};

std::string prepare_response_header(http_code code);

std::string get_page_content(webpage_t);
webpage_t get_default_page_stub();
response_t get_default_response (http_code code);


/* ****************************************************
 *                      Tags                          *
 **************************************************** */
namespace tag {
std::string doctype();

std::string gen_tag(const std::string& tag, const std::string& text);
std::string gen_inline_tag(const std::string& tag, const std::string& text);

std::string indent(const std::string& text, const std::string& ind = "\t");

std::string html(const std::string& text);
std::string head(const std::string& text);
std::string body(const std::string& text);
std::string meta(const web_meta_t& meta);
std::string title(const std::string& text);

std::string header(const std::string& text);
std::string center(const std::string& text);
std::string small(const std::string& text);


}

}
}



#endif /* SRC_MODULES_WEBSERVER_WEBPAGEGENERATOR_H_ */
