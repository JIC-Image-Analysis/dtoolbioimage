from ipywidgets import interactive, IntSlider, Output, HBox
from IPython.display import clear_output, display, display_png


def stack_viewer(stack):

    slider = IntSlider(min=0, max=100, step=1, continuous_update=False,
                       description='Z plane:', orientation='Vertical')

    o = Output()

    def update_viewer(change):
        with o:
            clear_output()
            display(stack[:, :, slider.value])


    slider.observe(update_viewer, 'value')
    update_viewer(None)

    return HBox([o, slider])

def simple_stack_viewer(stack):

    _, _, max_z = stack.shape

    slider = IntSlider(min=0, max=max_z, step=1, description='Z plane:')

    def show_z_plane(z):
        display_png(stack[:,:,z])
    
    return interactive(show_z_plane, z=slider)

