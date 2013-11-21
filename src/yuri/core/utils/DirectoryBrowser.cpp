/*!
 * @file 		DirectoryBrowser.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		16.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
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
		const std::string file = dit->path().string();
		const std::string fname = dit->path().filename().string();
		if (prefix.empty()) {
			paths.push_back(file);
		} else {
			if (fname.substr(0,prefix.size())==prefix) paths.push_back(file);
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


