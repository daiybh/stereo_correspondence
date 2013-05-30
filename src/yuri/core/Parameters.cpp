/*!
 * @file 		Parameters.cpp
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Parameters.h"
#include <iostream>
namespace yuri {

namespace core {
using boost::lexical_cast;

template<> shared_ptr<Callback> Parameter::get() const
{
	//if (type!=Callback_functionType) cbval.reset();
	return cbval;
}

Parameter & Parameter::operator= (yuri::ssize_t ival)
{
	type = IntegerType;
	this->ival = ival;
	return *this;
}
Parameter & Parameter::operator= (bool bval)
{
	type = BoolType;
	this->ival = bval;
	return *this;
}
Parameter & Parameter::operator= (double fval)
{
	type = FloatType;
	this->fval = fval;
	return *this;
}
Parameter & Parameter::operator= (std::string sval)
{
	type = StringType;
	this->sval = sval;
	return *this;
}

Parameter & Parameter::operator= (int ival)
{
	type = IntegerType;
	this->ival = ival;
	return *this;
}

Parameter & Parameter::operator= (float fval)
{
	type = FloatType;
	this->fval = fval;
	return *this;
}
Parameter & Parameter::operator= (const char *cval)
{
	type = StringType;
	this->sval =std::string(cval);
	return *this;
}

Parameter & Parameter::operator= (shared_ptr<Callback> cb)
{
	type = Callback_functionType;
	cbval = cb;
	return *this;
}
Parameter & Parameter::operator= (Parameter& par)
{
	type = par.type;
	switch(par.type) {
		case StringType: sval = par.get<std::string>(); break;
		case IntegerType: ival = par.get<yuri::ssize_t>(); break;
		case FloatType: fval = par.get<double>(); break;
		case BoolType: ival = par.get<bool>(); break;
		case Callback_functionType: cbval = par.get<shared_ptr<Callback> >(); break;
		case GroupType: {

			shared_ptr<Parameters> pr;
			BOOST_FOREACH(pr,par.parameters_vector) {
				shared_ptr<Parameters> par2(new Parameters(*pr));
				parameters_vector.push_back(par2);
			}
			break;
		}
		case NoneType:
		default:break;
	}
	return *this;
}


using namespace std;

shared_ptr<Parameter> Parameter::get_copy()
{
	shared_ptr<Parameter> p;
	switch (type) {
		case BoolType: p.reset(new Parameter(name,static_cast<bool>(ival))); break;
		case IntegerType: p.reset(new Parameter(name,ival)); break;
		case FloatType: p.reset(new Parameter(name,fval)); break;
		case StringType: p.reset(new Parameter(name,sval)); break;
		case Callback_functionType: p.reset(new Parameter(name,cbval)); break;
		case GroupType: {
			p.reset(new Parameter(name));
			shared_ptr<Parameters> par;
			BOOST_FOREACH(par,parameters_vector) {
				shared_ptr<Parameters> par2(new Parameters(*par));
				p->parameters_vector.push_back(par2);
			}
			break;
		}
		case NoneType:
		default: p.reset(new Parameter(name)); break;
	}
	return p;
}

Parameter& Parameter::operator[] (const std::string desc)
{
	description=desc;
	return *this;
}

Parameters& Parameter::push_group(std::string description0)
{
	type=GroupType;
	description=description0;
	shared_ptr<Parameters> par(new Parameters());
	parameters_vector.push_back(par);
	return *par;
}

Parameters& Parameter::operator[] (const yuri::size_t index)
{
	if (parameters_vector.size() < index+1) {
		throw OutOfRange("bad index "+lexical_cast<std::string>(index)+std::string(" out of ")+lexical_cast<std::string>(parameters_vector.size()));
	}
	if (!parameters_vector[index]) {
		throw Exception("Null pointer in params array, this should never happen");
	}
	return *(parameters_vector[index]);
}
Parameter::~Parameter()
{
//	cout << "Parameter " << name << " being deleted" << endl;
}



Parameters::Parameters() {
}

Parameters::Parameters(Parameters& p):
		max_input_pipes(-1), max_output_pipes(-1)
{
	*this=p;
/*	pair<std::string,shared_ptr<Parameter> > par;
	BOOST_FOREACH(par,p.params) {
		shared_ptr<Parameter> p = par.second->get_copy();
		this->params[par.first]=p;
	}*/
}

Parameters& Parameters::operator=(const Parameters& other)
{
	pair<std::string,shared_ptr<Parameter> > par;
	BOOST_FOREACH(par,other.params) {
		shared_ptr<Parameter> other = par.second->get_copy();
		this->params[par.first]=other;
	}
	this->description = other.description;
	return *this;

}

Parameters::~Parameters() {
//	cout << "Parameters being deleted" << endl;
}

Parameter& Parameters::operator[] (const std::string id)
{
	if (!is_defined(id)) params[id].reset(new Parameter(id));
	return *(params[id]);
}

bool Parameters::is_defined(std::string id)
{
	return !(params.find(id) == params.end());
}

void Parameters::merge(Parameters &p)
{
	pair<std::string,shared_ptr<Parameter> > par;
	BOOST_FOREACH(par,p.params) {
		(*this)[par.first]=*(par.second);
	}
}

void Parameters::set_description(std::string desc)
{
	description = desc;
}
string Parameters::get_description()
{
	return description;
}

void Parameters::set_max_pipes(yuri::sshort_t max_input, yuri::sshort_t max_output)
{
	max_input_pipes = max_input;
	max_output_pipes = max_output;
}

void Parameters::add_input_format(yuri::format_t format)
{
	input_formats.insert(format);
}
void Parameters::add_output_format(yuri::format_t format)
{
	output_formats.insert(format);
}

set<yuri::format_t> Parameters::get_input_formats()
{
	return input_formats;
}
set<yuri::format_t> Parameters::get_output_formats()
{
	return output_formats;
}

yuri::sshort_t Parameters::get_input_pipes()
{
	return max_input_pipes;
}
yuri::sshort_t Parameters::get_output_pipes()
{
	return max_output_pipes;
}

void Parameters::add_converter(yuri::format_t input, yuri::format_t output, int confidence, bool scaling)
{
	converters[make_pair(input,output)].reset(
			new Converter(input,output,"",confidence,scaling));
}

map<pair<yuri::format_t, yuri::format_t>, shared_ptr<Converter> > Parameters::get_converters()
{
	return converters;
}

}

}
