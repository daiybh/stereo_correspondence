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

}
}
}


#endif /* DIRECTORYBROWSER_H_ */
