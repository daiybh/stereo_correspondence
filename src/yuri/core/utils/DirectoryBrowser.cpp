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
#endif

namespace yuri {
namespace core {
namespace filesystem {

namespace {
bool has_prefix_suffix(const std::string& fname, const std::string& prefix, const std::string& suffix)
{
	if (!prefix.empty()) {
		if (fname.substr(0,prefix.size()) != prefix) return false;
	}
	if (!suffix.empty()) {
		if (suffix.size() > fname.size())  return false;
		if (fname.substr(fname.size()-suffix.size(),fname.size()) != suffix)  return false;
	}
	return true;
}
}

#ifdef HAVE_BOOST_FILESYSTEM
namespace {
std::vector<std::string> browse_files_impl(const std::string& path, const std::string& prefix, const std::string& suffix, bool dirsonly)
{
	std::vector<std::string> paths;
	boost::filesystem::path p(path);
	if (!boost::filesystem::exists(p) || !boost::filesystem::is_directory(p))
		return paths;
	for (boost::filesystem::directory_iterator dit(p);
						dit !=boost::filesystem::directory_iterator(); ++dit) {
		if (dirsonly && !boost::filesystem::is_directory(dit->path())) continue;
		const std::string file = dit->path().string();
		const std::string fname = dit->path().filename().string();
		if (!has_prefix_suffix(fname, prefix, suffix)) {
			continue;
		}
		paths.push_back(file);
	}
	return paths;
}
}

std::string get_directory(const std::string& filename)
{
	boost::filesystem::path f(filename);
	if (boost::filesystem::is_directory(f)) return filename;
	return f.parent_path().string();
}

std::string get_filename(const std::string& filename, bool with_extension)
{
	boost::filesystem::path f(filename);
	if (with_extension) return f.filename().string();
	return f.stem().string();
}

bool verify_path_exists(const std::string& path)
{
	boost::filesystem::path p(path);
	return boost::filesystem::exists(p);
}

bool create_directory(const std::string& dirname)
{
	return boost::filesystem::create_directories(dirname);
}

#elif defined YURI_POSIX
namespace {
std::vector<std::string> browse_files_impl(const std::string& path, const std::string& prefix, const std::string& suffix, bool dirsonly)
{
	std::vector<std::string> paths;
	dirent *dp;
	DIR *dfd = opendir(path.c_str());
	if(dfd) {
		while((dp = readdir(dfd)) ) {
			if (dirsonly && (dp->d_type != DT_DIR)) {
				continue;
			}
			std::string fname = dp->d_name;
			if (fname == "." || fname == "..")
					continue;
			if (!has_prefix_suffix(fname, prefix, suffix)) {
				continue;
			}
			paths.push_back(path+"/"+fname);
		}
		closedir(dfd);
	}
	return paths;
}
}

std::string get_directory(const std::string& filename)
{
	auto idx = filename.find_last_of('/');
	if (idx == std::string::npos) return {};
	return filename.substr(0, idx);
}
std::string get_filename(const std::string& filename, bool with_extension)
{
	auto idx = filename.find_last_of('/');
	if (idx == std::string::npos) return filename;
	if (with_extension) return filename.substr(idx+1);
	auto fname = filename.substr(idx+1);
	auto idx2 = fname.find_last_of('.');
	return fname.substr(0, idx2);
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
namespace {
std::vector<std::string> browse_files_impl(const std::string& path, const std::string& prefix, const std::string& suffix, bool dirsonly)
{
	return {};
}
}
std::string get_directory(const std::string& filename)
{
	auto idx = filename.find_last_of('/');
	if (idx == std::string::npos) return {};
	return filename.substr(0, idx);
}
std::string get_filename(const std::string& filename, bool with_extension)
{
	auto idx = filename.find_last_of('/');
	if (idx == std::string::npos) return filename;
	if (with_extension) return filename.substr(idx+1);
	auto fname = filename.substr(idx+1);
	auto idx2 = fname.find_last_of('.');
	return fname.substr(0, idx2);
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

std::vector<std::string> browse_files(const std::string& path, const std::string& prefix, const std::string& suffix)
{
	return browse_files_impl(path, prefix, suffix, false);
}
std::vector<std::string> browse_directories(const std::string& path, const std::string& prefix, const std::string& suffix)
{
	return browse_files_impl(path, prefix, suffix, true);
}


bool ensure_path_directory(const std::string& filename)
{
	const auto dir = get_directory(filename);
	return verify_path_exists(dir) || create_directory(dir);
}

}
}
}


