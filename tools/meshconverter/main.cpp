#include <arrrgh.hpp>
#include <shapeDescriptor/shapeDescriptor.h>

int main(int argc, const char** argv) {
    const std::string defaultExecutionDevice = ShapeDescriptor::isCUDASupportAvailable() ? "gpu" : "cpu";

    arrrgh::parser parser("meshconverter", "Generate RICI or spin images from an input object and dump them into a PNG file");
    const auto& inputFile = parser.add<std::string>(
            "input", "The location of the input OBJ model file.", '\0', arrrgh::Required, "");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message.", 'h', arrrgh::Optional, false);
    const auto& outputFile = parser.add<std::string>(
            "output", "The location of the PNG file to write to", '\0', arrrgh::Optional, "out.png");

    try
    {
        parser.parse(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if(showHelp.value())
    {
        return 0;
    }

    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::loadMesh(inputFile.value());
    ShapeDescriptor::writeMesh(mesh, outputFile.value());
    ShapeDescriptor::free(mesh);

}
