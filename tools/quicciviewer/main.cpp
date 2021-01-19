#include <arrrgh.hpp>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/read/QUICCIDescriptors.h>
#include <shapeDescriptor/utilities/print/QuicciDescriptor.h>

int main(int argc, const char** argv) {
    arrrgh::parser parser("quicciviewer", "Print QUICCI descriptors from an archive file to stdout.");
    const auto& inputArchive = parser.add<std::string>("input", "Location of the compressed raw descriptor dump file.", '\0', arrrgh::Required, "");
    const auto& imageIndex = parser.add<int>("image-index", "IOndex of the image to render.", '\0', arrrgh::Required, 0);
    const auto& showHelp = parser.add<bool>("help", "Show this help message.", 'h', arrrgh::Optional, false);

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

    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = ShapeDescriptor::read::QUICCIDescriptors(inputArchive.value());

    ShapeDescriptor::print::quicciDescriptor(descriptors.content[imageIndex.value()]);

    ShapeDescriptor::free::array(descriptors);
}

