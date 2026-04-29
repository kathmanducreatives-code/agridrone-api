-- AgriDrone Guardian Supabase schema
-- Run once in Supabase Dashboard -> SQL Editor, or apply through Supabase MCP.
--
-- This schema supports the current local FastAPI workflow:
-- ESP32/App -> FastAPI -> Supabase Storage/Postgres -> AI result back to app.
--
-- The policies below are intentionally development-open because the current
-- local backend is using a publishable key. Before public production, move
-- FastAPI to SUPABASE_SECRET_KEY and tighten anon policies.

create extension if not exists pgcrypto;

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'Crop-photos',
  'Crop-photos',
  false,
  10485760,
  array['image/jpeg', 'image/png', 'image/webp', 'application/octet-stream', 'text/plain']
)
on conflict (id) do update set
  name = excluded.name,
  public = excluded.public,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

drop table if exists public."Crop-Disease-Detection";

create table if not exists public.devices (
  id text primary key,
  thing_name text,
  hardware_type text,
  camera_type text,
  firmware_version text,
  field_id text,
  status text,
  last_seen_at text,
  last_reported_ip text,
  last_rssi integer,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.fields (
  id text primary key,
  name text,
  farm_name text,
  crop_type_default text,
  boundary_geojson jsonb,
  notes text,
  latest_sensor_snapshot jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.flights (
  flight_id text primary key,
  device_id text not null,
  field_id text not null,
  crop_type text not null default 'rice',
  crop_requested text,
  crop_warning text,
  operator_notes text,
  status text not null default 'awaiting_upload',
  upload_status text not null default 'pending',
  processing_status text not null default 'not_started',
  queue_job_id text,
  capture_interval_ms integer,
  model_version_id text,
  storage_folder text,
  summary jsonb,
  report jsonb,
  latest_sensor_snapshot jsonb,
  image_counts jsonb not null default '{}'::jsonb,
  created_at text,
  updated_at text,
  uploaded_at text,
  completed_at text,
  queued_at text,
  processing_started_at text,
  error_code text,
  error_message text
);

alter table public.flights add column if not exists storage_folder text;

create index if not exists idx_flights_device_id on public.flights (device_id);
create index if not exists idx_flights_field_id on public.flights (field_id);
create index if not exists idx_flights_created_at on public.flights (created_at desc);
create index if not exists idx_flights_status on public.flights (status);
create index if not exists idx_flights_storage_folder on public.flights (storage_folder);

create table if not exists public.flight_patches (
  id text primary key,
  flight_id text not null references public.flights(flight_id) on delete cascade,
  image_id text not null,
  patch_index integer not null default 0,
  captured_at text,
  storage_path text,
  storage_folder text,
  upload_status text not null default 'pending',
  analysis_status text not null default 'pending',
  gps jsonb,
  detections jsonb not null default '[]'::jsonb,
  primary_detection jsonb,
  content_type text,
  image_size jsonb,
  detection_count integer not null default 0,
  highest_severity text,
  crop_type text,
  crop_warning text,
  uploaded boolean not null default false,
  uploaded_at text,
  processed_at text,
  updated_at text,
  error_code text,
  error_message text,
  unique (flight_id, image_id)
);

alter table public.flight_patches add column if not exists storage_folder text;

create index if not exists idx_flight_patches_flight_id on public.flight_patches (flight_id);
create index if not exists idx_flight_patches_patch_index on public.flight_patches (flight_id, patch_index);
create index if not exists idx_flight_patches_analysis_status on public.flight_patches (analysis_status);
create index if not exists idx_flight_patches_storage_folder on public.flight_patches (storage_folder);

create table if not exists public.flight_reports (
  id text primary key,
  flight_id text not null references public.flights(flight_id) on delete cascade,
  top_disease text,
  highest_severity text,
  summary_json jsonb,
  report_json jsonb,
  generated_at text
);

create index if not exists idx_flight_reports_flight_id on public.flight_reports (flight_id);

create table if not exists public.inferences (
  inference_id text primary key,
  source_kind text not null default 'upload',
  source_label text,
  crop text not null default 'rice',
  crop_requested text,
  crop_warning text,
  status text not null default 'processing',
  request_content_type text,
  storage_path text,
  image_size jsonb not null default '{}'::jsonb,
  disease text,
  confidence double precision,
  severity text,
  prediction_type text,
  model text,
  model_kind text,
  artifact_label text,
  primary_detection jsonb,
  all_detections jsonb not null default '[]'::jsonb,
  top_predictions jsonb not null default '[]'::jsonb,
  firebase_saved boolean not null default false,
  supabase_saved boolean not null default false,
  latest_detection_saved boolean not null default false,
  record_saved boolean not null default true,
  backend_provider text,
  storage_mode text,
  created_at text,
  updated_at text,
  error_message text
);

create index if not exists idx_inferences_created_at on public.inferences (created_at desc);
create index if not exists idx_inferences_crop on public.inferences (crop);
create index if not exists idx_inferences_source_kind on public.inferences (source_kind);
create index if not exists idx_inferences_status on public.inferences (status);

create table if not exists public.latest_detections (
  id text primary key default 'latest',
  inference_id text,
  source_kind text,
  source_label text,
  crop text,
  disease text,
  confidence double precision,
  severity text,
  prediction_type text,
  model text,
  model_kind text,
  artifact_label text,
  storage_path text,
  image_size jsonb not null default '{}'::jsonb,
  timestamp bigint,
  primary_detection jsonb,
  all_detections jsonb not null default '[]'::jsonb,
  top_predictions jsonb not null default '[]'::jsonb,
  created_at text,
  updated_at timestamptz not null default now()
);

create table if not exists public.detection_history (
  id uuid primary key default gen_random_uuid(),
  inference_id text,
  source_kind text,
  source_label text,
  crop text,
  disease text,
  confidence double precision,
  severity text,
  prediction_type text,
  model text,
  model_kind text,
  artifact_label text,
  storage_path text,
  image_size jsonb not null default '{}'::jsonb,
  timestamp bigint,
  primary_detection jsonb,
  all_detections jsonb not null default '[]'::jsonb,
  top_predictions jsonb not null default '[]'::jsonb,
  created_at text,
  inserted_at timestamptz not null default now()
);

create index if not exists idx_detection_history_inserted_at on public.detection_history (inserted_at desc);
create index if not exists idx_detection_history_crop on public.detection_history (crop);

create table if not exists public.model_versions (
  id text primary key,
  crop_type text not null,
  model_name text not null,
  model_format text not null default 'onnx',
  storage_key text,
  version_label text,
  trained_at text,
  deployed_at text,
  is_active boolean not null default false,
  metrics jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_model_versions_crop_active on public.model_versions (crop_type, is_active);

create table if not exists public.device_events (
  id uuid primary key default gen_random_uuid(),
  device_id text,
  event_type text not null,
  severity text not null default 'info',
  message text,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_device_events_device_created on public.device_events (device_id, created_at desc);

insert into public.fields (id, name, farm_name, crop_type_default, notes)
values ('field-default', 'Default Test Field', 'AgriDrone Lab', 'rice', 'Default field for ESP32-CAM and Lab testing')
on conflict (id) do update set
  name = excluded.name,
  farm_name = excluded.farm_name,
  crop_type_default = excluded.crop_type_default,
  notes = excluded.notes,
  updated_at = now();

insert into public.devices (id, thing_name, hardware_type, camera_type, firmware_version, field_id, status)
values ('esp32-drone-01', 'agridrone', 'AI Thinker ESP32-CAM', 'OV2640/OV5640', 'local-ota', 'field-default', 'ready')
on conflict (id) do update set
  thing_name = excluded.thing_name,
  hardware_type = excluded.hardware_type,
  camera_type = excluded.camera_type,
  firmware_version = excluded.firmware_version,
  field_id = excluded.field_id,
  status = excluded.status,
  updated_at = now();

grant usage on schema public to anon, authenticated, service_role;
grant select on all tables in schema public to anon, authenticated;
grant select, insert, update, delete on all tables in schema public to service_role;
grant usage, select on all sequences in schema public to anon, authenticated, service_role;
alter default privileges in schema public grant select on tables to anon, authenticated;
alter default privileges in schema public grant select, insert, update, delete on tables to service_role;
alter default privileges in schema public grant usage, select on sequences to anon, authenticated, service_role;

alter table public.devices enable row level security;
alter table public.fields enable row level security;
alter table public.flights enable row level security;
alter table public.flight_patches enable row level security;
alter table public.flight_reports enable row level security;
alter table public.inferences enable row level security;
alter table public.latest_detections enable row level security;
alter table public.detection_history enable row level security;
alter table public.model_versions enable row level security;
alter table public.device_events enable row level security;

drop policy if exists agridrone_dev_all_devices on public.devices;
drop policy if exists agridrone_dev_all_fields on public.fields;
drop policy if exists agridrone_dev_all_flights on public.flights;
drop policy if exists agridrone_dev_all_flight_patches on public.flight_patches;
drop policy if exists agridrone_dev_all_flight_reports on public.flight_reports;
drop policy if exists agridrone_dev_all_inferences on public.inferences;
drop policy if exists agridrone_dev_all_latest_detections on public.latest_detections;
drop policy if exists agridrone_dev_all_detection_history on public.detection_history;
drop policy if exists agridrone_dev_all_model_versions on public.model_versions;
drop policy if exists agridrone_dev_all_device_events on public.device_events;

drop policy if exists agridrone_app_read_devices on public.devices;
create policy agridrone_app_read_devices on public.devices for select to anon, authenticated using (true);
drop policy if exists agridrone_app_read_fields on public.fields;
create policy agridrone_app_read_fields on public.fields for select to anon, authenticated using (true);
drop policy if exists agridrone_app_read_flights on public.flights;
create policy agridrone_app_read_flights on public.flights for select to anon, authenticated using (true);
drop policy if exists agridrone_app_read_flight_patches on public.flight_patches;
create policy agridrone_app_read_flight_patches on public.flight_patches for select to anon, authenticated using (true);
drop policy if exists agridrone_app_read_flight_reports on public.flight_reports;
create policy agridrone_app_read_flight_reports on public.flight_reports for select to anon, authenticated using (true);
drop policy if exists agridrone_app_read_inferences on public.inferences;
create policy agridrone_app_read_inferences on public.inferences for select to anon, authenticated using (true);
drop policy if exists agridrone_app_read_latest_detections on public.latest_detections;
create policy agridrone_app_read_latest_detections on public.latest_detections for select to anon, authenticated using (true);
drop policy if exists agridrone_app_read_detection_history on public.detection_history;
create policy agridrone_app_read_detection_history on public.detection_history for select to anon, authenticated using (true);
drop policy if exists agridrone_app_read_model_versions on public.model_versions;
create policy agridrone_app_read_model_versions on public.model_versions for select to anon, authenticated using (true);
drop policy if exists agridrone_app_read_device_events on public.device_events;
create policy agridrone_app_read_device_events on public.device_events for select to anon, authenticated using (true);

drop policy if exists agridrone_crop_photos_read on storage.objects;
drop policy if exists agridrone_crop_photos_insert on storage.objects;
drop policy if exists agridrone_crop_photos_update on storage.objects;

-- Realtime Postgres Changes for the field operator app. The app reads these
-- rows with the publishable key; FastAPI writes them with the service role key.
alter table public.flights replica identity full;
alter table public.flight_patches replica identity full;
alter table public.flight_reports replica identity full;
alter table public.inferences replica identity full;
alter table public.latest_detections replica identity full;

do $$
declare
  table_name text;
  realtime_tables text[] := array[
    'flights',
    'flight_patches',
    'flight_reports',
    'inferences',
    'latest_detections'
  ];
begin
  if not exists (
    select 1 from pg_publication where pubname = 'supabase_realtime'
  ) then
    execute 'create publication supabase_realtime';
  end if;

  foreach table_name in array realtime_tables loop
    if not exists (
      select 1
      from pg_publication_tables
      where pubname = 'supabase_realtime'
        and schemaname = 'public'
        and tablename = table_name
    ) then
      execute format('alter publication supabase_realtime add table public.%I', table_name);
    end if;
  end loop;
end $$;
