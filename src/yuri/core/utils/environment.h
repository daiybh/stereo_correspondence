/*!
 * @file 		environment.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		07.06.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef ENVIRON_H_
#define ENVIRON_H_

#include <string>
#include <vector>


namespace yuri {
namespace core {
namespace utils {

std::string get_environment_variable(const std::string& name, const std::string& def_value = std::string());
std::vector<std::string> get_environment_path(const std::string& name);

}
}
}






#endif /* ENVIRON_H_ */
