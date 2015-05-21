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

std::string get_directory(const std::string& filename)
{
	boost::filesystem::path f(filename);
	return f.parent_path().string();
}

bool verify_path_exists(const std::string& path)
{
	boost::filesystem::path p(path);
	return boost::filesystem::exists(p);
}

bool create_directory(const std::string& dirname)
{
	return boost::filesystem::create_directories(get_directory(dirname));
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

std::string get_directory(const std::string& filename)
{
	auto idx = filename.find_last_of('/');
	if (idx == std::string::npos) return {};
	return filename.substr(0, idx);
}

bool verify_path_exists(const std::string& path)
{
	return false;
}

bool create_directory(const std::string& dirname)
{
	return false;
}


#else
std::vector<std::string> browse_files(const std::string&, const std::string&)
{
	return {};
}
std::string get_directory(const std::string& filename)
{
	auto idx = filename.find_last_of('/');
	if (idx == std::string::npos) return {};
	return filename.substr(0, idx);
}

bool verify_path_exists(const std::string& path)
{
	return false;
}

bool create_directory(const std::string& dirname)
{
	return false;
}

#endif

bool ensure_path_directory(const std::string& filename)
{
	const auto dir = get_directory(filename);
	return verify_path_exists(dir) || create_directory(dir);
}

}
}
}


