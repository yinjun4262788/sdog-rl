import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import sys
# 用于GUI界面运行play.py
class DirectoryBrowser(tk.Toplevel):
    """自定义目录浏览器，支持单击选择和双击确认"""
    def __init__(self, parent, initialdir):
        super().__init__(parent)
        self.title("选择load_run目录")
        self.geometry("600x450")
        self.transient(parent)  # 依附于主窗口
        self.grab_set()  # 模态窗口
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        self.initialdir = os.path.abspath(initialdir)
        self.selected_dir = None
        
        # 创建UI
        self.create_widgets()
        
        # 初始化目录树
        self.populate_tree(self.initialdir)
        
    def create_widgets(self):
        """创建目录浏览器UI"""
        # 主框架
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 目录树
        self.tree = ttk.Treeview(main_frame)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 滚动条
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # 按钮框架
        btn_frame = ttk.Frame(self, padding="10")
        btn_frame.pack(fill=tk.X)
        
        # 确认按钮
        self.ok_btn = ttk.Button(btn_frame, text="确认", command=self.on_ok, state=tk.DISABLED)
        self.ok_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # 取消按钮
        cancel_btn = ttk.Button(btn_frame, text="取消", command=self.on_cancel)
        cancel_btn.pack(side=tk.RIGHT)
        
        # 绑定事件
        self.tree.bind('<<TreeviewSelect>>', self.on_select)  # 单击选择
        self.tree.bind('<Double-1>', self.on_double_click)  # 双击确认
        
    def populate_tree(self, path, parent=''):
        """填充目录树"""
        # 获取当前路径的目录名称
        path_name = os.path.basename(path) if path != '/' else '/'
        if not parent:
            # 根节点
            node = self.tree.insert('', 'end', text=path_name, values=(path,))
            self.tree.item(node, open=True)
        else:
            # 子节点
            try:
                node = self.tree.insert(parent, 'end', text=path_name, values=(path,))
            except:
                return
        
        # 加载子目录
        try:
            # 获取目录列表并排序
            entries = os.listdir(path)
            entries.sort()
            
            for entry in entries:
                entry_path = os.path.join(path, entry)
                # 检查是否是目录且有访问权限
                if os.path.isdir(entry_path) and os.access(entry_path, os.R_OK):
                    # 递归填充子目录（先不展开，节省资源）
                    self.populate_tree(entry_path, node)
                    
        except PermissionError:
            # 无权限访问的目录跳过
            pass
        except Exception as e:
            print(f"加载目录失败: {e}")
    
    def on_select(self, event):
        """单击选择目录"""
        # 获取选中的节点
        selected_items = self.tree.selection()
        if selected_items:
            item = selected_items[0]
            self.selected_dir = self.tree.item(item, 'values')[0]
            self.ok_btn.config(state=tk.NORMAL)  # 启用确认按钮
    
    def on_double_click(self, event):
        """双击确认选择"""
        self.on_select(event)
        if self.selected_dir:
            self.on_ok()
    
    def on_ok(self):
        """确认选择"""
        if self.selected_dir:
            self.destroy()  # 关闭对话框
    
    def on_cancel(self):
        """取消选择"""
        self.selected_dir = None
        self.destroy()

class SdogRunnerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SDog RL Runner")
        self.root.geometry("600x400")
        
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录 (假设目录结构为 project_root/legged_gym/scripts/this_script.py)
        self.project_root = os.path.dirname(os.path.dirname(current_dir))
        
        # 配置路径
        self.play_script_path = os.path.join(current_dir, "play.py")
        self.logs_dir = os.path.join(self.project_root, "logs")
        # 默认 logs_base_path (将在运行时根据 task 更新)
        self.logs_base_path = os.path.join(self.logs_dir, "rough_sdog")
        
        # 检查必要路径是否存在
        self.check_paths()
        
        # 创建UI
        self.create_widgets()
        
    def check_paths(self):
        """检查必要路径是否存在"""
        missing_paths = []
        
        if not os.path.exists(self.play_script_path):
            missing_paths.append(f"Play脚本: {self.play_script_path}")
        
        if not os.path.exists(self.logs_base_path):
            missing_paths.append(f"Logs目录: {self.logs_base_path}")
        
        if missing_paths:
            messagebox.showwarning("路径警告", "以下路径不存在，请检查：\n" + "\n".join(missing_paths))
    
    def create_widgets(self):
        """创建GUI组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Task选择区域
        task_frame = ttk.LabelFrame(main_frame, text="Task 设置", padding="10")
        task_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(task_frame, text="Task名称:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.task_var = tk.StringVar(value="sdog")
        self.task_entry = ttk.Entry(task_frame, textvariable=self.task_var, width=50)
        self.task_entry.grid(row=0, column=1, padx=10, pady=5)
        
        # Load_run选择区域
        load_run_frame = ttk.LabelFrame(main_frame, text="Load_run 设置", padding="10")
        load_run_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(load_run_frame, text="Load_run:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.load_run_var = tk.StringVar()
        self.load_run_entry = ttk.Entry(load_run_frame, textvariable=self.load_run_var, width=40)
        self.load_run_entry.grid(row=0, column=1, padx=10, pady=5)
        
        # 浏览按钮
        browse_btn = ttk.Button(load_run_frame, text="浏览", command=self.browse_load_run)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # 命令预览区域
        preview_frame = ttk.LabelFrame(main_frame, text="命令预览", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.command_text = tk.Text(preview_frame, height=4, wrap=tk.WORD)
        self.command_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 按钮区域
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 更新预览按钮
        update_btn = ttk.Button(btn_frame, text="更新预览", command=self.update_command_preview)
        update_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 运行按钮
        run_btn = ttk.Button(btn_frame, text="运行命令", command=self.run_command)
        run_btn.pack(side=tk.LEFT)
        
        # 清空按钮
        clear_btn = ttk.Button(btn_frame, text="清空", command=self.clear_fields)
        clear_btn.pack(side=tk.RIGHT)
        
        # 初始更新预览
        self.update_command_preview()
    
    def browse_load_run(self):
        """自定义浏览选择load_run目录"""
        # 根据当前 task 更新 logs_base_path
        task = self.task_var.get().strip()
        if task:
             self.logs_base_path = os.path.join(self.logs_dir, f"rough_{task}")
        
        if not os.path.exists(self.logs_base_path):
             messagebox.showwarning("路径不存在", f"找不到日志目录:\n{self.logs_base_path}\n请先运行训练生成日志，或检查Task名称是否正确。")
             return

        # 打开自定义目录浏览器
        dir_browser = DirectoryBrowser(self.root, self.logs_base_path)
        self.root.wait_window(dir_browser)  # 等待对话框关闭
        
        if dir_browser.selected_dir:
            # 提取目录名称（只保留Dec15_10-52-16_部分）
            dir_name = os.path.basename(dir_browser.selected_dir)
            self.load_run_var.set(dir_name)
            self.update_command_preview()
    
    def update_command_preview(self):
        """更新命令预览"""
        task = self.task_var.get().strip()
        load_run = self.load_run_var.get().strip()
        
        # 构建命令
        if task and load_run:
            command = (
                f"python {self.play_script_path} "
                f"--task={task} "
                f"--load_run={load_run}"
            )
        elif task:
            command = f"python {self.play_script_path} --task={task}"
        elif load_run:
            command = f"python {self.play_script_path} --load_run={load_run}"
        else:
            command = f"python {self.play_script_path}"
        
        # 更新文本框
        self.command_text.delete(1.0, tk.END)
        self.command_text.insert(tk.END, command)
    
    def run_command(self):
        """执行命令（移除确认弹窗）"""
        # 获取参数
        task = self.task_var.get().strip()
        load_run = self.load_run_var.get().strip()
        
        # 仅保留参数合法性校验
        if not task or not load_run:
            messagebox.showerror("错误", "请输入task和load_run参数！")
            return
        
        # 构建完整命令
        command = [
            "python",
            self.play_script_path,
            f"--task={task}",
            f"--load_run={load_run}"
        ]
        
        try:
            # 直接执行命令（无确认弹窗）
            #messagebox.showinfo("信息", "命令开始执行...\n请查看终端输出")
            
            # 在新的终端中执行命令（Ubuntu）
            subprocess.Popen(
                ['x-terminal-emulator', '-e', 'bash', '-c', ' '.join(command) + '; read -p "按Enter键关闭窗口..."'],
                cwd=os.path.dirname(self.play_script_path)
            )
            
        except Exception as e:
            messagebox.showerror("执行错误", f"命令执行失败：\n{str(e)}")
    
    def clear_fields(self):
        """清空输入字段"""
        self.task_var.set("sdog")  # 保留默认值
        self.load_run_var.set("")
        self.update_command_preview()

def main():
    """主函数"""
    # 检查是否在Ubuntu系统
    if not os.path.exists('/etc/lsb-release'):
        messagebox.showwarning("系统警告", "此工具专为Ubuntu系统设计，可能在其他系统上无法正常工作")
    
    # 创建主窗口
    root = tk.Tk()
    app = SdogRunnerGUI(root)
    
    # 设置窗口图标（可选）
    try:
        root.iconphoto(False, tk.PhotoImage(file='/usr/share/icons/ubuntu-logo.png'))
    except:
        pass
    
    # 运行主循环
    root.mainloop()

if __name__ == "__main__":
    # 检查Python环境
    if sys.version_info < (3, 6):
        messagebox.showerror("Python版本错误", "需要Python 3.6或更高版本")
    else:
        main()