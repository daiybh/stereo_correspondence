/*!
 * @file 		Parameters.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_
#include <string>
#include <map>
#include <set>
#include <vector>
#include <yuri/exception/Exception.h>
#include <yuri/exception/OutOfRange.h>

#include "Callback.h"
namespace yuri {

namespace core {

using namespace yuri::exception;

enum _type {
	StringType,
	IntegerType,
	FloatType,
	BoolType,
	Callback_functionType,
	GroupType,
	NoneType
};

class EXPORT Converter {
private:
	Converter():confidence(0),scaling(false) {};
public:
	Converter (yuri::format_t input, yuri::format_t output, std::string id, int confidence, bool scaling = false):
		format(input,output), id(id), confidence(confidence), scaling(scaling) {}
	std::pair<yuri::format_t, yuri::format_t> format;
	std::string id;
	int confidence;
	bool scaling;

};
class Parameters;

class EXPORT Parameter {
public:
	Parameter(std::string name):name(name),type(NoneType),ival(0),fval(0.0) {}
	Parameter(std::string name,yuri::ssize_t ival):name(name),type(IntegerType),ival(ival),fval(0.0) {}
	Parameter(std::string name,bool bval):name(name),type(BoolType),ival(bval),fval(0.0) {}
	Parameter(std::string name,double fval):name(name),type(FloatType),ival(0),fval(fval) {}
	Parameter(std::string name, std::string sval):name(name),type(StringType),sval(sval),ival(0),fval(0.0) {}
	Parameter(std::string name,shared_ptr<Callback> cb):name(name),type(Callback_functionType),ival(0),fval(0.0),cbval(cb) {}

	std::string 			name;
	std::string 			description;
	_type 					type;

	std::string 			sval;
	yuri::ssize_t 			ival;
	double 					fval;
	pCallback				cbval;
	std::vector<pParameters >
							parameters_vector;

	Parameter & 			operator= (yuri::ssize_t ival);
	Parameter & 			operator= (bool bval);
	Parameter & 			operator= (double fval);
	Parameter & 			operator= (std::string sval);

	Parameter & 			operator= (int ival);
	Parameter & 			operator= (float fval);
	Parameter & 			operator= (const char *cval);

	Parameter & 			operator= (Parameter& par);

	Parameter & 			operator= (shared_ptr<Callback> cb);

	Parameter& 				operator[] (const std::string id);
	Parameters& 			operator[] (const yuri::size_t index);

	template <typename T> T get() const;

	Parameters& 			push_group(std::string description="");

	shared_ptr<Parameter> 	get_copy();
	~Parameter();
};

template <typename T> T Parameter::get() const
{
	try {
	switch (type) {
		case BoolType:
		case IntegerType: return lexical_cast<T>(ival);
		case FloatType: return lexical_cast<T>(fval);
		case StringType: return lexical_cast<T>(sval);
		default: return static_cast<T> (0);
	}
	}
	catch (bad_lexical_cast& ) {
		throw Exception(std::string("Bad cast in parameter ")+name);
	}
}

class EXPORT Parameters {
public:
								Parameters();
								Parameters(Parameters& p);
	virtual 					~Parameters();
	Parameters& 				operator=(const Parameters& other);

	Parameter& 					operator[] (const std::string id);
	bool 						is_defined(std::string id);
	std::map<std::string,shared_ptr<Parameter> >
								params;
	void 						merge(Parameters&p);
	void 						set_description(std::string desc);
	std::string 				get_description();
	void 						set_max_pipes(yuri::sshort_t max_input, yuri::sshort_t  max_output);
	void 						add_input_format(yuri::format_t format);
	void 						add_output_format(yuri::format_t format);
	std::set<yuri::format_t> 	get_input_formats();
	std::set<yuri::format_t> 	get_output_formats();
	yuri::sshort_t 				get_input_pipes();
	yuri::sshort_t 				get_output_pipes();
	void 						add_converter(yuri::format_t input, yuri::format_t output,
				int confidence=0, bool scaling=false);
	std::map<std::pair<yuri::format_t, yuri::format_t>, shared_ptr<Converter> >
								get_converters();
protected:
	std::string 				description;
	// Maximal number of input/output pipes. -1 for unlimited.
	int 						max_input_pipes;
	int							max_output_pipes;
	std::set<yuri::format_t > 	input_formats;
	std::set<yuri::format_t> 	output_formats;
	std::map<std::pair<yuri::format_t, yuri::format_t>, shared_ptr<Converter> >
								converters;
};

}

}

#endif /* PARAMETERS_H_ */
