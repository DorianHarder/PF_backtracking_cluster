cluster_roundnes = []
for cl in clusters:
    coordinates = []
    # gpd.GeoSeries(cl).plot(ax = ax, color=colors[color],markersize=5)
    alpha_shape = alphashape.alphashape(MultiPoint(cl), 1.)
    if alpha_shape.area > 0:
        cluster_roundnes.append(alpha_shape.length/alpha_shape.area)
    else:
        cluster_roundnes.append(1)

    if cluster_roundnes[0] > 0.8:
        gpd.GeoSeries(alpha_shape).plot(ax=ax, color='green', markersize=5)
    else:
        gpd.GeoSeries(alpha_shape).plot(ax=ax, color='red', markersize=5)

    color += 1