import numpy as np
import cv2 
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def plot_points(points, colors = None):
    '''
    plot pointcloud
    When using o3d.geometry.PointCloud;
    in: points: data
    When using np.array;
    in: points: np.array([[x1,y1,z1],[x2,y2,z2], ......[xn,yn,zn]])
        colors(opt.): np.array([[b1,g1,r1], [b2,g2,r2], .....[bn,gn,rn]])
    '''
    if type(points) is o3d.geometry.PointCloud:
        pcd = points

    elif type(points) is np.ndarray:
        # remove np.nan
        # (H, W, 3) のような形状を想定
        is_point = np.where(np.logical_not(np.isnan(points)))
        points = points[is_point]
        # reshape
        points = points.reshape((int(points.size/3),3))
        # convert to py3d.PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    
        if not colors is None:
            # colors も (H, W, 3) を想定
            colors = np.array(colors[is_point], np.uint8)
            colors = colors.reshape((1, int(colors.size/3), 3))
            colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors[0])
  
    o3d.visualization.draw_geometries([pcd])


def cluster_points_dbscan(points, eps=0.01, min_samples=10):
    """
    点群データに対してDBSCANクラスタリングを実行します。

    in:
        points (np.ndarray): (N, 3)形状の点群データ。
        eps (float): DBSCANのepsパラメータ（近傍とみなす最大距離）。
        min_samples (int): DBSCANのmin_samplesパラメータ（クラスターを形成する最小点数）。
    out:
        labels (np.ndarray): (N,)形状のクラスターラベル。ノイズは-1。
    """
    print(f"DBSCANを実行中... (eps={eps}, min_samples={min_samples})")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"推定されたクラスター数: {n_clusters}")
    print(f"ノイズとして検出された点群数: {n_noise}")
    
    return labels


def create_circle_lineset(center, radius=1, normal=[0, 0, 1], resolution=32, color=[1, 0, 0]):
    """
    指定された中心と法線を持つ円のLineSetを生成します。
    
    in:
        center (np.array): 円の中心座標 [x, y, z]
        radius (float): 円の半径
        normal (list): 円が乗る平面の法線ベクトル (デフォルトはZ軸)
        resolution (int): 円を近似するための線分の数
        color (list): 円の色 [R, G, B] (0-1)
    out:
        o3d.geometry.LineSet: 円形状のラインセット
    """
    # 法線ベクトルを正規化
    normal = np.array(normal) / np.linalg.norm(normal)
    
    # 法線に直交する2つのベクトル (u, v) を見つける
    # (簡易的な方法: normal が [0, 0, 1] でない場合も対応)
    if np.allclose(normal, [0, 0, 1]) or np.allclose(normal, [0, 0, -1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
        u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v) # vも正規化

    points = []
    for i in range(resolution + 1):
        theta = 2.0 * np.pi * i / resolution
        # 円周上の点を計算: P = Center + R*cos(t)*u + R*sin(t)*v
        p = center + radius * (np.cos(theta) * u + np.sin(theta) * v)
        points.append(p)
    
    lines = []
    for i in range(resolution):
        lines.append([i, i + 1])
        
    colors = [color] * len(lines)
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def find_knot_target_on_top_cluster(points, labels, circle_radius=0.05):
    """
    クラスタリングされた点群から「箱の上の紐」を特定し、
    X座標最大の点（ターゲット）と、その点に描画する円を返します。

    in:
        points (np.array): (N, 3) のクラスタリング対象となった点群
        labels (np.array): (N,) のDBSCANラベル (ノイズは-1)
        circle_radius (float): 描画する円の半径
    out:
        target_point (np.array): X座標最大の点 [x, y, z]
        circle_geom (o3d.geometry.LineSet): 描画用の円
        top_cluster_pcd (o3d.geometry.PointCloud): 抽出された「箱の上の紐」の点群
    """
    
    # ノイズ(-1)を除いたユニークなクラスターIDを取得
    unique_labels = set(labels)
    unique_labels.discard(-1)
    
    if not unique_labels:
        print("有効なクラスターが見つかりませんでした。")
        return None, None, None

    # 各クラスターの平均Z座標を計算
    top_cluster_id = -1
    max_mean_z = -np.inf
    
    for label_id in unique_labels:
        cluster_points = points[labels == label_id]
        mean_z = np.mean(cluster_points[:, 2]) # Z座標の平均
        
        if mean_z > max_mean_z:
            max_mean_z = mean_z
            top_cluster_id = label_id
            
    print(f"「箱の上の紐」クラスターID: {top_cluster_id} (平均Z座標: {max_mean_z:.4f})")
    
    # 「箱の上の紐」の点群データを取得
    top_cluster_points = points[labels == top_cluster_id]
    
    # X座標が最大となる点を検索
    max_x_index = np.argmax(top_cluster_points[:, 0]) # X座標 (インデックス0)
    target_point = top_cluster_points[max_x_index]
    
    print(f"希望結び目位置 (X最大): {target_point}")
    
    # ターゲット点を中心に円を生成 (XY平面、つまり法線=[0,0,1])
    circle_geom = create_circle_lineset(
        center=target_point, 
        radius=circle_radius, 
        normal=[0, 0, 1], # Z軸に垂直な円 (XY平面)
        color=[1, 0, 0] # 赤色
    )
    
    # 可視化のために、このクラスターの点群オブジェクトも作成
    top_cluster_pcd = o3d.geometry.PointCloud()
    top_cluster_pcd.points = o3d.utility.Vector3dVector(top_cluster_points)
    top_cluster_pcd.paint_uniform_color([0, 1, 0]) # 「箱の上の紐」を緑色に
    
    return target_point, circle_geom, top_cluster_pcd


# --- メイン処理 ---

# 1. 保存した .npz ファイルを読み込む
input_filename = './data/my_filtered_pointcloud.npz'
print(f"{input_filename} を読み込んでいます...")
try:
    data = np.load(input_filename)

    # 2. 保存時に指定した名前で配列を取り出す
    loaded_points = data['points']
    loaded_colors = data['colors']

    print(f"読み込み成功。")

    # 3. 読み込んだデータでプロット (元の紐の点群)
    print("元の（フィルタリング済み）紐の点群を表示します...")
    plot_points(loaded_points, loaded_colors)

    # 4. DBSCANによるクラスタリング処理

    # 4-1. 点群データを (N, 3) の形式に変換する
    points_flat = loaded_points.reshape(-1, 3)
    valid_points = points_flat[~np.isnan(points_flat).any(axis=1)]
    
    if valid_points.shape[0] == 0:
        print("有効な点群が見つかりませんでした。")
        raise ValueError("No valid points to cluster.")

    print(f"クラスタリング対象の点群数: {valid_points.shape[0]}")

    # 4-2. DBSCAN関数を呼び出す
    labels = cluster_points_dbscan(valid_points, eps=10, min_samples=30)

    # 4-3. 「箱の上の紐」を特定し、ターゲット（円）を計算
    
    # 描画するジオメトリを格納するリスト
    geometries_to_draw = []
    
    # (オプション) DBSCANで色分けした全クラスターを背景として表示する場合
    # ノイズ除去
    non_noise_indices = np.where(labels != -1)[0]
    points_clustered = valid_points[non_noise_indices]
    labels_clustered = labels[non_noise_indices]

    if points_clustered.shape[0] > 0:
        # クラスターごとに色を割り当てる
        max_label = labels_clustered.max()
        if max_label >= 0:
            cmap = plt.get_cmap("tab10", max_label + 1)
            norm_labels = labels_clustered / max_label if max_label > 0 else labels_clustered.astype(float)
            colors_rgb = cmap(norm_labels)[:, :3]
            
            pcd_clustered = o3d.geometry.PointCloud()
            pcd_clustered.points = o3d.utility.Vector3dVector(points_clustered)
            pcd_clustered.colors = o3d.utility.Vector3dVector(colors_rgb)
            
            geometries_to_draw.append(pcd_clustered)
            print("DBSCANクラスタリング結果（色分け）を準備しました。")


    # 新しい関数を呼び出して、ターゲット点と円を取得
    target_pos, target_circle, top_pcd = find_knot_target_on_top_cluster(
        valid_points, 
        labels,
        circle_radius=10# 円の半径 (例: 3cm)
    )

    # 4-4. 結果の可視化
    if target_pos is not None:
        print(f"最終的な希望結び目位置: {target_pos}")
        
        # (オプション) もし「箱の上の紐」だけをハイライトしたい場合
        # geometries_to_draw.clear() # 他のクラスターを消す
        # geometries_to_draw.append(top_pcd) # 緑色の「上の紐」だけ追加

        # ターゲットの円を追加
        geometries_to_draw.append(target_circle) 
        
        print("ターゲット位置に円を描画して表示します...")
        o3d.visualization.draw_geometries(geometries_to_draw)
        
    else:
        print("ターゲット点の計算に失敗しました。")
        # クラスタリング結果だけを表示（もしあれば）
        if geometries_to_draw:
            print("クラスタリング結果のみ表示します...")
            o3d.visualization.draw_geometries(geometries_to_draw)


except FileNotFoundError:
    print(f"エラー: ファイル '{input_filename}' が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {e}")