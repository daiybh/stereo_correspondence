/*
 * Parameters.h
 *
 *  Created on: Jul 24, 2010
 *      Author: neneko
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

using namespace std;
using namespace yuri::exception;

using boost::shared_ptr;

enum _type {
	StringType,
	IntegerType,
	FloatType,
	BoolType,
	Callback_functionType,
	GroupType,
	NoneType
};

class Converter {
private:
	Converter() {};
public:
	Converter (yuri::format_t input, yuri::format_t output, string id, int confidence, bool scaling = false):
		format(input,output), id(id), confidence(confidence), scaling(scaling) {}
	pair<yuri::format_t, yuri::format_t> format;
	string id;
	int confidence;
	bool scaling;

};
class Parameters;

class Parameter {
public:
	Parameter(string name):name(name),type(NoneType) {}
	Parameter(string name,yuri::ssize_t ival):name(name),type(IntegerType),ival(ival) {}
	Parameter(string name,bool bval):name(name),type(BoolType),ival(bval) {}
	Parameter(string name,double fval):name(name),type(FloatType),fval(fval) {}
	Parameter(string name,string sval):name(name),type(StringType),sval(sval) {}
	Parameter(string name,shared_ptr<Callback> cb):name(name),type(Callback_functionType),cbval(cb) {}

	string name, description;
	_type type;

	string sval;
	yuri::ssize_t ival;
	double fval;
	shared_ptr<yuri::config::Callback> cbval;
	vector<shared_ptr<Parameters> > parameters_vector;

	Parameter & operator= (yuri::ssize_t ival);
	Parameter & operator= (bool bval);
	Parameter & operator= (double fval);
	Parameter & operator= (string sval);

	Parameter & operator= (int ival);
	Parameter & operator= (float fval);
	Parameter & operator= (const char *cval);

	Parameter & operator= (Parameter& par);

	Parameter & operator= (shared_ptr<Callback> cb);

	Parameter& operator[] (const string id);
	Parameters& operator[] (const yuri::size_t index) throw (Exception);

	template <typename T> T get() throw (Exception);

	Parameters& push_group(string description="");

	shared_ptr<Parameter> get_copy();
	~Parameter();
};

template <typename T> T Parameter::get() throw (Exception)
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
		throw Exception(string("Bad cast in parameter ")+name);
	}
}

class Parameters {
public:
	Parameters();
	Parameters(Parameters& p);
	virtual ~Parameters();
	EXPORT Parameters& operator=(const Parameters& other);

	EXPORT Parameter& operator[] (const string id) throw (Exception);
	EXPORT bool is_defined(string id);
	map<string,shared_ptr<Parameter> > params;
	EXPORT void merge(Parameters&p);
	EXPORT void set_description(string desc);
	EXPORT string get_description();
	EXPORT void set_max_pipes(yuri::sshort_t max_input, yuri::sshort_t  max_output);
	EXPORT void add_input_format(yuri::format_t format);
	EXPORT void add_output_format(yuri::format_t format);
	EXPORT set<yuri::format_t> get_input_formats();
	EXPORT set<yuri::format_t> get_output_formats();
	EXPORT yuri::sshort_t get_input_pipes();
	EXPORT yuri::sshort_t get_output_pipes();
	void add_converter(yuri::format_t input, yuri::format_t output, int confidence=0, bool scaling=false);
	EXPORT map<pair<yuri::format_t, yuri::format_t>, shared_ptr<Converter> > get_converters();
protected:
	string description;
	// Maximal number of input/output pipes. -1 for unlimited.
	int max_input_pipes, max_output_pipes;
	set<yuri::format_t > input_formats;
	set<yuri::format_t> output_formats;
	map<pair<yuri::format_t, yuri::format_t>, shared_ptr<Converter> > converters;
};

}

}

#endif /* PARAMETERS_H_ */
