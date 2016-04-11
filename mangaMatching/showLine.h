#pragma once

#include "mangaShow.h"
mangaShow ms;

namespace mangaMatching {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::IO;

	/// <summary>
	/// showLine 的摘要
	/// </summary>
	public ref class showLine : public System::Windows::Forms::Form
	{
	public:
		showLine(void)
		{
			InitializeComponent();
			//
			//TODO:  在此加入建構函式程式碼
			//
		}

	protected:
		/// <summary>
		/// 清除任何使用中的資源。
		/// </summary>
		~showLine()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::PictureBox^  showLine_pictureBox;
	private: System::Windows::Forms::GroupBox^  lineCheck_groupBox;
	protected:

	private:
		/// <summary>
		/// 設計工具所需的變數。
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// 此為設計工具支援所需的方法 - 請勿使用程式碼編輯器
		/// 修改這個方法的內容。
		/// </summary>
		void InitializeComponent(void)
		{
			this->showLine_pictureBox = (gcnew System::Windows::Forms::PictureBox());
			this->lineCheck_groupBox = (gcnew System::Windows::Forms::GroupBox());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->showLine_pictureBox))->BeginInit();
			this->SuspendLayout();
			// 
			// showLine_pictureBox
			// 
			this->showLine_pictureBox->AllowDrop = true;
			this->showLine_pictureBox->BackColor = System::Drawing::Color::Black;
			this->showLine_pictureBox->Location = System::Drawing::Point(13, 13);
			this->showLine_pictureBox->Name = L"showLine_pictureBox";
			this->showLine_pictureBox->Size = System::Drawing::Size(900, 836);
			this->showLine_pictureBox->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->showLine_pictureBox->TabIndex = 0;
			this->showLine_pictureBox->TabStop = false;
			this->showLine_pictureBox->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &showLine::showLine_pictureBox_DragDrop);
			this->showLine_pictureBox->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &showLine::showLine_pictureBox_DragEnter);
			// 
			// lineCheck_groupBox
			// 
			this->lineCheck_groupBox->Location = System::Drawing::Point(920, 13);
			this->lineCheck_groupBox->Name = L"lineCheck_groupBox";
			this->lineCheck_groupBox->Size = System::Drawing::Size(252, 836);
			this->lineCheck_groupBox->TabIndex = 1;
			this->lineCheck_groupBox->TabStop = false;
			// 
			// showLine
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 12);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1184, 861);
			this->Controls->Add(this->lineCheck_groupBox);
			this->Controls->Add(this->showLine_pictureBox);
			this->Name = L"showLine";
			this->Text = L"showLine";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->showLine_pictureBox))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion
	private: System::Void showLine_pictureBox_DragDrop(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e) {
				 if (e->Data->GetDataPresent(DataFormats::FileDrop)){
					 array<System::String^>^files = (array<System::String^>^)e->Data->GetData(DataFormats::FileDrop);
					 System::String ^imgExt = ".jpg|.png|.bmp|.jpeg|.gif";
					 try{
						 System::String ^ext = Path::GetExtension(files[0]);
						 if (imgExt->IndexOf(ext) >= 0){
							 char *fileName = (char*)System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(files[0]).ToPointer();
							 ms = mangaShow(fileName);

							 LARGE_INTEGER start_t, end_t, freq;
							 QueryPerformanceFrequency(&freq);
							 QueryPerformanceCounter(&start_t);

							 ms.vector_curves();
							 ms.remove_dump_by_ROI();
							 //ms.rng_curves_color();
							 //ms.set_curves_drawable();
							 //ms.draw_curves();
							 ms.topol_curves();
							 ms.link_adjacent();
							 ms.draw_topol();

							 QueryPerformanceCounter(&end_t);
							 std::cout << ((double)end_t.QuadPart - (double)start_t.QuadPart) / freq.QuadPart << std::endl;

							 this->showLine_pictureBox->Image = ms.get_canvas_Bitmap();

							 this->lineCheck_groupBox->Controls->Clear();
							 std::vector<bool> curves_drawable = ms.get_curves_drawable();
							 for (int i = 0; i < curves_drawable.size(); ++i){
								 CheckBox ^cb = gcnew System::Windows::Forms::CheckBox();
								 cb->AutoSize = true;
								 cb->Size = System::Drawing::Size(77, 16);
								 cb->Location = System::Drawing::Point(7 + cb->Size.Width * (i % 3), i / 3 * cb->Size.Height + 16);
								 cb->Name = L"line" + (i > 9 ? "" : "0") + i;
								 cb->Text = L"line" + (i > 9 ? "" : "0") + i;
								 cb->TabIndex = i + 2;
								 cb->Checked = curves_drawable[i];
								 cb->UseVisualStyleBackColor = true;
								 cb->CheckStateChanged += gcnew System::EventHandler(this, &showLine::cb_CheckStateChanged);
								 this->lineCheck_groupBox->Controls->Add(cb);
							 }
						 }
					 }
					 catch (System::Exception^ ex){
						 MessageBox::Show(ex->Message);
						 return;
					 }
				 }
	}
	private: System::Void showLine_pictureBox_DragEnter(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e) {
				 if (e->Data->GetDataPresent(DataFormats::FileDrop))
					 e->Effect = DragDropEffects::Copy;
				 else
					 e->Effect = DragDropEffects::None;
	}
	private: System::Void cb_CheckStateChanged(System::Object^  sender, System::EventArgs^  e) {
				 std::vector<bool> curves_drawable = ms.get_curves_drawable();
				 for (int i = 0; i < curves_drawable.size(); ++i){
					 ms.set_curves_drawable(i, ((CheckBox ^)this->lineCheck_groupBox->Controls[i])->Checked);
				 }
				 ms.draw_curves();
				 this->showLine_pictureBox->Image = ms.get_canvas_Bitmap();

	}
	};
}
