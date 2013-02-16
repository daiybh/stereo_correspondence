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
#include <boost/foreach.hpp>
#include <yuri/exception/Exception.h>
#include <yuri/exception/OutOfRange.h>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include "Callback.h"
namespace yuri {

namespace config {

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
	Converter() {};
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
	Parameter(std::string name):name(name),type(NoneType) {}
	Parameter(std::string name,yuri::ssize_t ival):name(name),type(IntegerType),ival(ival) {}
	Parameter(std::string name,bool bval):name(name),type(BoolType),ival(bval) {}
	Parameter(std::string name,double fval):name(name),type(FloatType),fval(fval) {}
	Parameter(std::string name, std::string sval):name(name),type(StringType),sval(sval) {}
	Parameter(std::string name,shared_ptr<Callback> cb):name(name),type(Callback_functionType),cbval(cb) {}

std::string name, description;
	_type type;

std::string sval;
	yuri::ssize_t ival;
	double fval;
	shared_ptr<yuri::config::Callback> cbval;
	std::vector<shared_ptr<Parameters> > parameters_vector;

	Parameter & operator= (yuri::ssize_t ival);
	Parameter & operator= (bool bval);
	Parameter & operator= (double fval);
	Parameter & operator= (std::string sval);

	Parameter & operator= (int ival);
	Parameter & operator= (float fval);
	Parameter & operator= (const char *cval);

	Parameter & operator= (Parameter& par);

	Parameter & operator= (shared_ptr<Callback> cb);

	Parameter& operator[] (const std::string id);
	Parameters& operator[] (const yuri::size_t index);

	template <typename T> T get();

	Parameters& push_group(std::string description="");

	shared_ptr<Parameter> get_copy();
	~Parameter();
};

template <typename T> T Parameter::get()
{
	try {
	switch (type) {
		case BoolType:
		case IntegerType: return boost::lexical_cast<T>(ival);
		case FloatType: return boost::lexical_cast<T>(fval);
		case StringType: return boost::lexical_cast<T>(sval);
		default: return static_cast<T> (0);
	}
	}
	catch (boost::bad_lexical_cast) {
		throw Exception(std::string("Bad cast in parameter ")+name);
	}
}

class EXPORT Parameters {
public:
	Parameters();
	Parameters(Parameters& p);
	virtual ~Parameters();
	Parameters& operator=(const Parameters& other);

	Parameter& operator[] (const std::string id);
	bool is_defined(std::string id);
	std::map<std::string,shared_ptr<Parameter> > params;
	void merge(Parameters&p);
	void set_description(std::string desc);
	std::string get_description();
	void set_max_pipes(yuri::sshort_t max_input, yuri::sshort_t  max_output);
	void add_input_format(yuri::format_t format);
	void add_output_format(yuri::format_t format);
	std::set<yuri::format_t> get_input_formats();
	std::set<yuri::format_t> get_output_formats();
	yuri::sshort_t get_input_pipes();
	yuri::sshort_t get_output_pipes();
	void add_converter(yuri::format_t input, yuri::format_t output, int confidence=0, bool scaling=false);
	std::map<std::pair<yuri::format_t, yuri::format_t>, shared_ptr<Converter> > get_converters();
protected:
	std::string description;
	// Maximal number of input/output pipes. -1 for unlimited.
	int max_input_pipes, max_output_pipes;
	std::set<yuri::format_t > input_formats;
	std::set<yuri::format_t> output_formats;
	std::map<std::pair<yuri::format_t, yuri::format_t>, shared_ptr<Converter> > converters;
};

}

}

#endif /* PARAMETERS_H_ */
