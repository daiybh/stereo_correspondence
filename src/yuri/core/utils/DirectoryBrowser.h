/*!
 * @file 		DirectoryBrowser.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		16.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef DIRECTORYBROWSER_H_
#define DIRECTORYBROWSER_H_
#include <string>
#include <vector>

namespace yuri {
namespace core {
namespace filesystem {

std::vector<std::string> browse_files(const std::string& path, const std::string& prefix = std::string());


/*!
 * Returns directory component from specified filename
 * @param filename
 * @return
 */
std::string get_directory(const std::string& filename);

/*!
 * Returns filename component from specified filename
 * @param filename
 * @param with_extension set to false to strip extension
 * @return
 */
std::string get_filename(const std::string& filename, bool with_extension = true);
/*!
 * Verifies that a specified path exists in the filesystem
 * @param path Filesystem path
 * @return true iff the path represents a valid object in the filesystem.
 */
bool verify_path_exists(const std::string& path);

/*!
 * Creates a directory
 * @param dirname
 * @return true if the directory was created successfully.
 */
bool create_directory(const std::string& dirname);

/*!
 * Verifies the directory for specified filename exists, and creates it if it doesn't
 * @param filename
 * @return true if the directory exists sfter the method call.
 */
bool ensure_path_directory(const std::string& filename);

}
}
}


#endif /* DIRECTORYBROWSER_H_ */
