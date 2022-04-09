
#include "Context.hpp"
#include "Renderer.hpp"

int main(void)
{
    Context context;
    Renderer renderer(context);

    bool running = true;
    while (running)
    {
        running = renderer.render();
    }

    return 0;
}