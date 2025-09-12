
def thin_lens_zi(f, z0):
    return 1 / (1/f - 1/z0)

def aperture_diameter(f, N):
    return f / N