#include "Context.hpp"
#include "Rasterizer.hpp"
#include "Raytracer.hpp"

int main(void)
{
    Context context;
    //Rasterizer graphicsApp(context);
    Raytracer graphicsApp(context);

    bool running = true;
    while (running)
    {
        running = graphicsApp.render();
    }

    return 0;
}
