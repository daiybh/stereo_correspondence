/*
 * DirectoryBrowser.h
 *
 *  Created on: 16.9.2013
 *      Author: neneko
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
