"""Live display of the channels in a track observation."""

from matplotlib import pyplot as plt


class TrackViewDisplay:
    def __init__(self, view_names, channel_counts, *, mask_label='Track boundary'):
        columns = max(channel_counts)
        self._figure, axes = plt.subplots(
            len(view_names), columns, squeeze=False,
            figsize=(3.5 * columns, 3.4 * len(view_names)), layout='constrained',
            num='Track observations',
        )
        self._canvas = self._figure.canvas
        self._status = self._figure.suptitle('', animated=self._canvas.supports_blit)
        self._images = []
        self._background = None
        if self._canvas.supports_blit:
            self._canvas.mpl_connect('draw_event', self._cache_background)

        self._axes = []
        for row, name, count in zip(axes, view_names, channel_counts):
            active_axes = row[:count]
            self._axes.append(active_axes)
            for axis in row[count:]:
                axis.remove()
            for axis, channel in zip(active_axes, (mask_label, 'Distance to car')):
                axis.set_title(f'{name}\n{channel}')
                axis.set_axis_off()

    @property
    def is_open(self):
        return plt.fignum_exists(self._figure.number)

    def update(self, views, step, steering):
        self._status.set_text(f'Step {step}    Steering {steering:+.2f}')
        if self._images:
            for images, view in zip(self._images, views):
                for image, channel in zip(images, view):
                    image.set_data(channel)
        else:
            for axes, view in zip(self._axes, views):
                self._images.append([
                    axis.imshow(channel, cmap='gray', vmin=0, vmax=1,
                                interpolation='nearest', origin='upper',
                                animated=self._canvas.supports_blit)
                    for axis, channel in zip(axes, view)
                ])
            active_axes = [axis for row in self._axes for axis in row]
            self._figure.colorbar(self._images[0][0], ax=active_axes, label='Channel value')
            self._background = None
            plt.show(block=False)

        if self._background is None:
            self._canvas.draw_idle()
        else:
            self._canvas.restore_region(self._background)
            self._draw_frame()
            self._canvas.blit(self._figure.bbox)
        self._canvas.flush_events()

    def _cache_background(self, _event):
        # A full draw, including after a resize, replaces the cached layout.
        self._background = self._canvas.copy_from_bbox(self._figure.bbox)
        self._draw_frame()

    def _draw_frame(self):
        for images in self._images:
            for image in images:
                self._figure.draw_artist(image)
        self._figure.draw_artist(self._status)

    def close(self):
        plt.close(self._figure)
