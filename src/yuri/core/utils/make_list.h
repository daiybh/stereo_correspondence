/*!
 * @file 		make_list.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		01.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef MAKE_LIST_H_
#define MAKE_LIST_H_

namespace yuri {
namespace core {
namespace utils {
template<class T>
std::string make_list(const std::vector<T>& src, const std::string& separator =", ")
{
	std::string out;
	for (const auto& x: src) {
		if (!out.empty()) {
			out += separator;
		}
		out += lexical_cast<std::string>(x);
	}
	return out;
}

template<class Str, class T>
void print_list(Str& os, const std::vector<T>& src, const std::string& separator =", ")
{
	bool first = true;
	for (const auto& x: src) {
		if (!first) {
			os << separator;
		} else {
			first = false;
		}
		os << lexical_cast<std::string>(x);
	}
}

}
}
}



#endif /* MAKE_LIST_H_ */
