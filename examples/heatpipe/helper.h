#ifndef HELPER_H
#define HELPER_H

#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include<vector>
#include <iterator>

struct Parameters
{
    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;
    double xr, yr, zr;
    double xcold, ycold, zcold;
    double sample_x, sample_y, sample_z;
    double sample_lx, sample_ly, sample_lz;
    

    Parameters():
        xmin(0.0), xmax(0.0),
        ymin(0.0), ymax(0.0),
        zmin(0.0), zmax(0.0),
        xr(0.0), yr(0.0), zr(0.0),
        xcold(0.0), ycold(0.0), zcold(0.0),
        sample_x(0.0), sample_y(0.0), sample_z(0.0),
        sample_lx(0.0), sample_ly(0.0), sample_lz(0.0)
    {}
};

namespace helper_detail
{

inline std::string trim(const std::string & text)
{
    std::string::size_type first(0);
    while (first < text.size() &&
           std::isspace(static_cast<unsigned char>(text[first])))
        ++first;

    std::string::size_type last(text.size());
    while (last > first &&
           std::isspace(static_cast<unsigned char>(text[last - 1])))
        --last;

    return text.substr(first, last - first);
}

} // namespace helper_detail

// Reads lines in the form "xmin = 3.52814". Empty lines and lines
// beginning with '#' are ignored.
inline void readKeyValues(const std::string & fileName, Parameters & p)
{
    std::ifstream input(fileName.c_str());
    if (!input)
        throw std::runtime_error("Could not open parameter file: " + fileName);

    bool found[18] = {
        false, false, false,
        false, false, false,
        false, false, false,
        false, false, false,
        false, false, false,
        false, false, false
    };
    std::string line;
    unsigned int lineNumber(0);

    while (std::getline(input, line))
    {
        ++lineNumber;
        line = helper_detail::trim(line);
        if (line.empty() || line[0] == '#')
            continue;

        const std::string::size_type separator(line.find('='));
        if (separator == std::string::npos)
            throw std::runtime_error(
                "Missing '=' on line " + std::to_string(lineNumber));

        const std::string key(helper_detail::trim(line.substr(0, separator)));
        const std::string valueText(
            helper_detail::trim(line.substr(separator + 1)));
        if (key.empty() || valueText.empty())
            throw std::runtime_error(
                "Invalid key/value on line " + std::to_string(lineNumber));

        // Arrays are parsed separately by readTValues() and readPValues().
        if (key == "T" || key == "P")
            continue;

        std::string::size_type parsedCharacters(0);
        double value;
        try
        {
            value = std::stod(valueText, &parsedCharacters);
        }
        catch (const std::exception &)
        {
            throw std::runtime_error(
                "Invalid number on line " + std::to_string(lineNumber));
        }

        if (!helper_detail::trim(valueText.substr(parsedCharacters)).empty())
            throw std::runtime_error(
                "Unexpected text after value on line " +
                std::to_string(lineNumber));

        if (key == "xmin")
        {
            p.xmin = value;
            found[0] = true;
        }
        else if (key == "xmax")
        {
            p.xmax = value;
            found[1] = true;
        }
        else if (key == "ymin")
        {
            p.ymin = value;
            found[2] = true;
        }
        else if (key == "ymax")
        {
            p.ymax = value;
            found[3] = true;
        }
        else if (key == "zmin")
        {
            p.zmin = value;
            found[4] = true;
        }
        else if (key == "zmax")
        {
            p.zmax = value;
            found[5] = true;
        }
        else if (key == "zr")
        {
            p.zr = value;
            found[6] = true;
        }
        else if (key == "xr")
        {
            p.xr = value;
            found[7] = true;
        }
        else if (key == "yr")
        {
            p.yr = value;
            found[8] = true;
        }

        else if (key == "zcold")
        {
            p.zcold = value;
            found[9] = true;
        }
        else if (key == "xcold")
        {
            p.xcold = value;
            found[10] = true;
        }
        else if (key == "ycold")
        {
            p.ycold = value;
            found[11] = true;
        }
        else if (key == "sample_x")
        {
            p.sample_x = value;
            found[12] = true;
        }
        else if (key == "sample_y")
        {
            p.sample_y = value;
            found[13] = true;
        }
        else if (key == "sample_z")
        {
            p.sample_z = value;
            found[14] = true;
        }
        else if (key == "sample_lx")
        {
            p.sample_lx = value;
            found[15] = true;
        }
        else if (key == "sample_ly")
        {
            p.sample_ly = value;
            found[16] = true;
        }
        else if (key == "sample_lz")
        {
            p.sample_lz = value;
            found[17] = true;
        }
        else
        {
            throw std::runtime_error(
                "Unknown key '" + key + "' on line " +
                std::to_string(lineNumber));
        }
    }

    const char * keys[18] = {
        "xmin", "xmax", "ymin", "ymax", "zmin", "zmax",
        "zr", "xr", "yr", "zcold", "xcold", "ycold",
        "sample_x", "sample_y", "sample_z",
        "sample_lx", "sample_ly", "sample_lz"
    };
    for (unsigned int i(0); i < 12; ++i)
        if (!found[i])
            throw std::runtime_error(
                "Missing required key '" + std::string(keys[i]) + "'");
}

inline void printKeyValues(const Parameters & p)
{
    std::cout << "xmin = " << p.xmin << '\n';
    std::cout << "xmax = " << p.xmax << '\n';
    std::cout << "ymin = " << p.ymin << '\n';
    std::cout << "ymax = " << p.ymax << '\n';
    std::cout << "zmin = " << p.zmin << '\n';
    std::cout << "zmax = " << p.zmax << '\n';
    std::cout << "zr = " << p.zr << '\n';
    std::cout << "yr = " << p.yr << '\n';
    std::cout << "xr = " << p.xr << '\n';
    std::cout << "zcold = " << p.zcold << '\n';
    std::cout << "ycold = " << p.ycold << '\n';
    std::cout << "xcold = " << p.xcold << '\n';
    std::cout << "sample_x = " << p.sample_x << '\n';
    std::cout << "sample_y = " << p.sample_y << '\n';
    std::cout << "sample_z = " << p.sample_z << '\n';
    std::cout << "sample_lx = " << p.sample_lx << '\n';
    std::cout << "sample_ly = " << p.sample_ly << '\n';
    std::cout << "sample_lz = " << p.sample_lz << '\n';
}


inline std::vector<double> readArrayValues(
    const std::string& fileName,
    const std::string& arrayName)
{
    if (arrayName.empty())
        throw std::runtime_error("Array name cannot be empty");

    std::ifstream input(fileName.c_str());
    if (!input)
        throw std::runtime_error("Could not open file: " + fileName);

    const std::string text(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>()
    );

    std::size_t position = 0;
    std::size_t openingBrace = std::string::npos;

    // Find a standalone array name followed by '=' and '{'.
    while ((position = text.find(arrayName, position)) != std::string::npos)
    {
        const bool validLeft =
            position == 0 ||
            (!std::isalnum(static_cast<unsigned char>(text[position - 1])) &&
             text[position - 1] != '_');

        std::size_t cursor = position + arrayName.size();

        while (cursor < text.size() &&
               std::isspace(static_cast<unsigned char>(text[cursor])))
            ++cursor;

        if (validLeft && cursor < text.size() && text[cursor] == '=')
        {
            ++cursor;

            while (cursor < text.size() &&
                   std::isspace(static_cast<unsigned char>(text[cursor])))
                ++cursor;

            if (cursor < text.size() && text[cursor] == '{')
            {
                openingBrace = cursor;
                break;
            }
        }

        ++position;
    }

    if (openingBrace == std::string::npos)
        throw std::runtime_error(
            "Could not find " + arrayName + " = {...} in " + fileName);

    const std::size_t closingBrace = text.find('}', openingBrace + 1);
    if (closingBrace == std::string::npos)
        throw std::runtime_error(
            "Missing closing '}' for " + arrayName + " values");

    const std::string valueText = text.substr(
        openingBrace + 1,
        closingBrace - openingBrace - 1
    );

    std::stringstream stream(valueText);
    std::vector<double> values;

    while (true)
    {
        stream >> std::ws;

        if (stream.peek() == std::char_traits<char>::eof())
            break;

        double value;
        if (!(stream >> value))
            throw std::runtime_error(
                "Invalid number in " + arrayName + " array");

        values.push_back(value);

        stream >> std::ws;

        if (stream.peek() == std::char_traits<char>::eof())
            break;

        char comma;
        if (!(stream >> comma) || comma != ',')
            throw std::runtime_error(
                "Expected ',' between values in " + arrayName + " array");
    }

    if (values.empty())
        throw std::runtime_error(arrayName + " array is empty");

    return values;
}

inline std::vector<double> readTValues(const std::string& fileName)
{
    return readArrayValues(fileName, "T");
}

inline std::vector<double> readPValues(const std::string& fileName)
{
    return readArrayValues(fileName, "P");
}

inline void printTValues(const std::vector<double>& T)
{
    std::cout << "T = {";

    for (std::size_t i = 0; i < T.size(); ++i)
    {
        if (i > 0)
            std::cout << ", ";

        std::cout << T[i];
    }

    std::cout << "}\n";
}

inline void printPValues(const std::vector<double>& P)
{
    std::cout << "P = {";

    for (std::size_t i = 0; i < P.size(); ++i)
    {
        if (i > 0)
            std::cout << ", ";

        std::cout << P[i];
    }

    std::cout << "}\n";
}

#endif
