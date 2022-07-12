#include "Context.hpp"
#include "Renderer.hpp"
#include "Raytracer.hpp"

int main(void)
{
    Context context;
    //Renderer graphicsApp(context);
    Raytracer graphicsApp(context);

    bool running = true;
    while (running)
    {
        running = graphicsApp.render();
    }

    return 0;
}
