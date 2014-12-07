/*
 * WebPageGenerator.cpp
 *
 *  Created on: Dec 7, 2014
 *      Author: neneko
 */

#include "WebPageGenerator.h"
#include "yuri/version.h"
#include <boost/regex.hpp>
namespace yuri {
namespace webserver {

namespace {
const std::map<http_code, std::string> common_codes = {
		{http_code::continue_ , "Continue"},
		{http_code::ok, "OK"},
		{http_code::created, "Created"},
		{http_code::accepted, "Accepted"},
		{http_code::no_content, "No content"},
		{http_code::partial, "Partial"},
		{http_code::moved, "Moved Permanently"},
		{http_code::found, "Found"},
		{http_code::see_other, "See Other"},
		{http_code::not_modified, "Not Modified"},
		{http_code::bad_request, "Bad Request"},
		{http_code::unauthorized, "Unauthorized"},
		{http_code::forbidden, "Forbidden"},
		{http_code::not_found, "Not Found"},
		{http_code::gone, "Gone"},
		{http_code::server_error, "Internal Server Error"},
		{http_code::service_unavailable, "Service Unavailable"}
	};

	const std::string yuri_version_full = std::string{"yuri-"}+yuri_version;

	std::string get_footer()
	{
		return tag::center(tag::small(yuri_version_full));
	}
	int http_code_to_int(http_code code) {
		return static_cast<int>(code);
	}
}


std::string get_code_name(http_code code)
{
	auto it = common_codes.find(code);
	if (it == common_codes.end()) return "UNKNOWN";
	return it->second;
}
std::string prepare_response_header(http_code code)
{
	return "HTTP/1.1 " + std::to_string(http_code_to_int(code)) + " " + get_code_name(code);
}

std::string get_page_content(webpage_t page)
{
	std::string metas;
	for (const auto&meta: page.metas) {
		metas+=tag::meta(meta);
	}
	return tag::doctype() +
			tag::html(
				tag::head(
					tag::title(page.title) +
					metas
					) +
				tag::body(
					page.body +
					get_footer()
				)
			);

}
webpage_t get_default_page_stub()
{
	webpage_t page {
		std::string{"Yuri ("}+yuri_version+")",
		{{"Generator",yuri_version_full}},
		{},
		true
	};
	return page;
}
response_t get_default_response (http_code code, const std::string& reason)
{
	response_t response {code,{},{}};
	webpage_t page = get_default_page_stub();
	page.title=get_code_name(code)+" ("+yuri_version_full+")";
	page.body=tag::header(std::to_string(http_code_to_int(code))+" "+get_code_name(code));
	if (!reason.empty()) {
		page.body += tag::line_break()+reason+tag::line_break();
	}

	response.data=get_page_content(std::move(page));
	return response;

}


url_t parse_url(const std::string& uri, const std::string& host)
{
	url_t url;
	url.host = host;
	boost::regex url_line("^([^?&#]+)");

	boost::smatch what;
	auto start = uri.cbegin();
	const auto end = uri.cend();
	if (regex_search(start, end, what, url_line, boost::match_default)) {
		url.path = std::string(what[1].first, what[1].second);
		auto next = what[0].second;
		boost::regex param_line ("[?&]([^&=#]+)(=([^&#]+))?");
		boost::sregex_iterator i(next, end, param_line, boost::match_default);
		boost::sregex_iterator j;
		while (i != j) {
			const auto& res = *i;
			const std::string param_name (res[1].first, res[1].second);
			url.params[param_name]=std::string(res[3].first,res[3].second);
//			for (auto p:res) {
//				std::cout << "match: " <<std::string(p.first,p.second) << "\n";
//			}
			++i;
		}


	}


	return url;
}


namespace tag {

std::string doctype()
{
	return "<!DOCTYPE html>\n";
}
std::string gen_tag(const std::string& tag, const std::string& text)
{
	return "<"+tag+">\n"+indent(text)+"</"+tag+">\n";
}
std::string gen_inline_tag(const std::string& tag, const std::string& text)
{
	return "<"+tag+">"+text+"</"+tag+">\n";
}
std::string gen_empty_tag(const std::string& tag)
{
	return "</"+tag+">\n";
}

std::string indent(const std::string& text, const std::string& ind)
{
	std::string out = ind;
	out.reserve(text.size()+2*ind.size());
	for (const auto&c:text) {
		out.push_back(c);
		if (c=='\n') {
			out.append(ind.begin(),ind.end());
		}

	}
	return out;
}

std::string html(const std::string& text)
{
	return gen_tag("html",text);
}
std::string head(const std::string& text)
{
	return gen_tag("head",text);
}
std::string body(const std::string& text)
{
	return gen_tag("body",text);
}
std::string meta(const web_meta_t& meta)
{
	return "<meta name=\""+meta.name+"\" content=\""+meta.content+"\"/>\n";
}

std::string title(const std::string& text)
{
	return gen_inline_tag("title",text);
}

std::string header(const std::string& text)
{
	return gen_inline_tag("h1",text);
}
std::string center(const std::string& text)
{
	return gen_inline_tag("center",text);
}
std::string small(const std::string& text)
{
	return gen_inline_tag("small",text);
}
std::string line_break()
{
	return gen_empty_tag("br");
}


}


}
}


