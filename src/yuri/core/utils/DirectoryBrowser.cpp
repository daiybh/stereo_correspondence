/*
 * DirectoryBrowser.cpp
 *
 *  Created on: 16.9.2013
 *      Author: neneko
 */

#include "DirectoryBrowser.h"
#include "yuri/core/utils/platform.h"
#ifdef HAVE_BOOST_FILESYSTEM
#include <boost/filesystem.hpp>
#elif defined YURI_POSIX
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <iostream>
#endif


namespace yuri {
namespace core {
namespace filesystem {

#ifdef HAVE_BOOST_FILESYSTEM
std::vector<std::string> browse_files(const std::string& path, const std::string& prefix)
{
	std::vector<std::string> paths;
	boost::filesystem::path p(path);
	if (!boost::filesystem::exists(p) || !boost::filesystem::is_directory(p))
		return paths;
	for (boost::filesystem::directory_iterator dit(p);
						dit !=boost::filesystem::directory_iterator(); ++dit) {
		const auto& file = dit->path();
		if (prefix.empty()) {
			paths.push_back(file.string());
		} else {
			const auto& filename = file.filename().string();
			if (filename.substr(0,prefix.size())==prefix) paths.push_back(file.string());
		}
	}
	return paths;
}
#elif defined YURI_POSIX
std::vector<std::string> browse_files(const std::string& path, const std::string& prefix)
{
	std::vector<std::string> paths;
	dirent *dp;
	DIR *dfd = opendir(path.c_str());
	if(dfd) {
		while((dp = readdir(dfd)) ) {
			std::string fname = dp->d_name;
			if (prefix.empty()) {
				paths.push_back(path+"/"+fname);
			} else {
				if (fname.substr(0,prefix.size())==prefix) paths.push_back(path+"/"+fname);
			}
		}
		closedir(dfd);
	}
	return paths;
}
#else
std::vector<std::string> browse_files(const std::string&, const std::string&)
{
	return {};
}
#endif
}
}
}


