import numpy as np
import cv2 # .npzの読み込み自体には不要だが、もし追加処理するなら
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
    # DBSCANを初期化して実行
    db = DBSCAN(eps=eps, min_samples=min_samples)
    # fit_predictは (N, 3) の入力を期待する
    labels = db.fit_predict(points)
    
    # 結果のサマリーを表示
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"推定されたクラスター数: {n_clusters}")
    print(f"ノイズとして検出された点群数: {n_noise}")
    
    return labels

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
    # print(f"Points shape: {loaded_points.shape}")
    # print(f"Colors shape: {loaded_colors.shape}")

    # 3. 読み込んだデータでプロット (元の紐の点群)
    print("元の（フィルタリング済み）紐の点群を表示します...")
    plot_points(loaded_points, loaded_colors)

    # 4. DBSCANによるクラスタリング処理

    # 4-1. 点群データを (N, 3) の形式に変換する
    # loaded_points (H, W, 3) から np.nan を除外し (N, 3) の配列を作成
    points_flat = loaded_points.reshape(-1, 3)
    valid_points = points_flat[~np.isnan(points_flat).any(axis=1)]
    
    if valid_points.shape[0] == 0:
        print("有効な点群が見つかりませんでした。")
        raise ValueError("No valid points to cluster.")

    print(f"クラスタリング対象の点群数: {valid_points.shape[0]}")

    # 4-2. DBSCAN関数を呼び出す
    # (注意) epsとmin_samplesは、点群のスケール（メートル単位か、ミリメートル単位か）や
    # 密度によって大きく変わります。適切な値に調整してください。
    # ここでは仮に eps=0.01 (1cm), min_samples=10 とします。
    labels = cluster_points_dbscan(valid_points, eps=10, min_samples=30)

    # 4-3. 結果の可視化 (ノイズ除去と色分け)
    
    # ノイズ（-1）を除去したインデックスを取得
    non_noise_indices = np.where(labels != -1)[0]
    
    # ノイズ除去後の点群とラベル
    points_clustered = valid_points[non_noise_indices]
    labels_clustered = labels[non_noise_indices]
    
    if points_clustered.shape[0] == 0:
        print("ノイズ除去後、点群が残りませんでした。epsが小さすぎるか、min_samplesが大きすぎる可能性があります。")
    else:
        print(f"ノイズ除去後の点群数: {points_clustered.shape[0]}")
        
        # クラスターごとに色を割り当てる
        max_label = labels_clustered.max()
        if max_label < 0:
            print("有効なクラスターが見つかりませんでした。")
        else:
            # matplotlibのカラーマップを使って色を生成 (RGB, 0-1)
            # クラスター数が少ない場合は 'tab10' や 'Set1' が見やすい
            cmap = plt.get_cmap("tab10", max_label + 1)
            
            # ラベルを0からmax_labelの範囲に正規化して色をマッピング
            # (max_labelが0の場合のゼロ除算を回避)
            norm_labels = labels_clustered / max_label if max_label > 0 else labels_clustered.astype(float)
            colors_rgb = cmap(norm_labels) # 0-1 の RGBA
            colors_rgb = colors_rgb[:, :3] # RGBのみ抽出
            
            # Open3DのPointCloudオブジェクトを作成
            pcd_clustered = o3d.geometry.PointCloud()
            pcd_clustered.points = o3d.utility.Vector3dVector(points_clustered)
            pcd_clustered.colors = o3d.utility.Vector3dVector(colors_rgb)
            
            print("クラスタリング結果（ノイズ除去・色分け済み）を表示します...")
            o3d.visualization.draw_geometries([pcd_clustered])

except FileNotFoundError:
    print(f"エラー: ファイル '{input_filename}' が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {e}")