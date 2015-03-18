#ifdef YURI_STD_TO_STRING_MISSING
#include <cstdio>
#include <string>
namespace std {
inline std::string to_string( int value )
{
	char buf[100];
	std::sprintf(buf, "%d", value);
	return buf;
}
inline std::string to_string( long value )
{
	char buf[100];
	std::sprintf(buf, "%ld", value);
	return buf;
}
inline std::string to_string( long long value )
{
	char buf[100];
	std::sprintf(buf, "%lld", value);
	return buf;
}
inline std::string to_string( unsigned value )
{
	char buf[100];
	std::sprintf(buf, "%u", value);
	return buf;
}
inline std::string to_string( unsigned long value )
{
	char buf[100];
	std::sprintf(buf, "%lu", value);
	return buf;
}
inline std::string to_string( unsigned long long value )
{
	char buf[100];
	std::sprintf(buf, "%llu", value);
	return buf;
}
inline std::string to_string( float value )
{
	char buf[100];
	std::sprintf(buf, "%f", value);
	return buf;
}
inline std::string to_string( double value )
{
	char buf[100];
	std::sprintf(buf, "%f", value);
	return buf;
}
inline std::string to_string( long double value ) 
{
	char buf[100];
	std::sprintf(buf, "%Lf", value);
	return buf;
}
}
#endif

#ifdef YURI_STD_STOUL_MISSING
#include <cstdlib>
#include <string>
namespace std {
inline unsigned long stoul(const std::string& s) 
{ 
	return std::strtoul(s.c_str(), nullptr, 10); 
}
}
#endif