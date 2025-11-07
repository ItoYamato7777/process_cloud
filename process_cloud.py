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
        # (H, W, 3) のような形状を (H*W, 3) に変形
        points_shape = points.shape
        if len(points_shape) == 3 and points_shape[2] == 3:
            # (H, W, 3) -> (H*W, 3)
            points_flat = points.reshape(-1, 3)
        elif len(points_shape) == 2 and points_shape[1] == 3:
            # すでに (N, 3) 形式
            points_flat = points
        else:
            print(f"サポート外のNumpy配列形状です: {points_shape}")
            return

        pcd = o3d.geometry.PointCloud()

        if not colors is None:
            # colors も points と同じ形状を想定
            colors_shape = colors.shape
            if len(colors_shape) == 3 and colors_shape[2] == 3:
                colors_flat = colors.reshape(-1, 3)
            elif len(colors_shape) == 2 and colors_shape[1] == 3:
                colors_flat = colors
            else:
                print(f"サポート外のNumpy配列形状です (Colors): {colors_shape}")
                return

            # points の NaN に基づいてフィルタリング
            valid_mask = ~np.isnan(points_flat).any(axis=1)
            
            points_valid = points_flat[valid_mask]
            colors_valid = colors_flat[valid_mask]
            
            pcd.points = o3d.utility.Vector3dVector(points_valid)

            # BGR -> RGB 変換 (uint8想定)
            if colors_valid.size > 0:
                colors_rgb = cv2.cvtColor(colors_valid.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2RGB) / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors_rgb[0])
            
        else:
            # colors がない場合
            valid_mask = ~np.isnan(points_flat).any(axis=1)
            points_valid = points_flat[valid_mask]
            
            pcd.points = o3d.utility.Vector3dVector(points_valid)
  
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


def find_knot_target_on_top_cluster(points, labels, sphere_radius=0.05): # 引数名をsphere_radiusに変更
    """
    クラスタリングされた点群から「箱の上の紐」を特定し、
    X座標最大の点（ターゲット）と、その点に描画する球体を返します。

    in:
        points (np.array): (N, 3) のクラスタリング対象となった点群
        labels (np.array): (N,) のDBSCANラベル (ノイズは-1)
        sphere_radius (float): 描画する球体の半径
    out:
        target_point (np.array): X座標最大の点 [x, y, z]
        sphere_geom (o3d.geometry.TriangleMesh): 描画用の球体
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

    if top_cluster_points.shape[0] == 0:
        print(f"クラスター {top_cluster_id} に点が見つかりません。")
        return None, None, None
    
    # X座標が最大となる点を検索
    max_x_index = np.argmax(top_cluster_points[:, 0]) # X座標 (インデックス0)
    target_point = top_cluster_points[max_x_index]
    
    print(f"希望結び目位置 (X最大): {target_point}")
    
    # ターゲット点を中心に球体を生成
    # o3d.geometry.TriangleMesh.create_mesh_sphere を使用
    sphere_geom = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere_geom.translate(target_point) # 中心をターゲット点に移動
    sphere_geom.paint_uniform_color([1, 0, 0]) # 赤色に設定
    
    # 可視化のために、このクラスターの点群オブジェクトも作成
    top_cluster_pcd = o3d.geometry.PointCloud()
    top_cluster_pcd.points = o3d.utility.Vector3dVector(top_cluster_points)
    top_cluster_pcd.paint_uniform_color([0, 1, 0]) # 「箱の上の紐」を緑色に
    
    return target_point, sphere_geom, top_cluster_pcd # 返り値をsphere_geomに変更


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

    # 4-3. 「箱の上の紐」を特定し、ターゲット（球体）を計算
    
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
            # cmapの色数を指定
            cmap = plt.get_cmap("tab10") 
            colors_rgb = np.zeros((labels_clustered.shape[0], 3))
            
            # 各ラベルID (0, 1, 2, ...) を cmap のインデックス (0-9) にマッピング
            for i, label_id in enumerate(np.unique(labels_clustered)):
                indices = np.where(labels_clustered == label_id)
                # cmap の色数で剰余を取り、循環させる
                color_index = label_id % cmap.N 
                colors_rgb[indices] = cmap(color_index)[:3]

            pcd_clustered = o3d.geometry.PointCloud()
            pcd_clustered.points = o3d.utility.Vector3dVector(points_clustered)
            pcd_clustered.colors = o3d.utility.Vector3dVector(colors_rgb)
            
            geometries_to_draw.append(pcd_clustered)
            print("DBSCANクラスタリング結果（色分け）を準備しました。")


    # 新しい関数を呼び出して、ターゲット点と球体を取得
    # 変数名をtarget_sphereに変更
    target_pos, target_sphere, top_pcd = find_knot_target_on_top_cluster(
        valid_points, 
        labels,
        sphere_radius=3 # 球体の半径 (例: 3cm)
    )

    # 4-4. 結果の可視化
    if target_pos is not None and target_sphere is not None: # target_sphereのチェックを追加
        print(f"最終的な希望結び目位置: {target_pos}")
        
        # (オプション) もし「箱の上の紐」だけをハイライトしたい場合
        # geometries_to_draw.clear() # 他のクラスターを消す
        # geometries_to_draw.append(top_pcd) # 緑色の「上の紐」だけ追加

        # ターゲットの球体を追加
        geometries_to_draw.append(target_sphere) 
        
        print("ターゲット位置に球体を描画して表示します...")
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